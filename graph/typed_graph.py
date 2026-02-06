from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from graph.core import Graph, MemSpace, Node, SourceLoc
from graph.nodes.op import OpNode
from ops import select_op
from static_validator import _canonical_op_name, _infer_op_metadata


class TypedGraphError(ValueError):
    """Raised for malformed typed .graph files."""


def _parse_value(raw: str) -> Any:
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part.strip()) for part in inner.split(",")]
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1].strip()
        if not inner:
            return {}
        out: Dict[str, Any] = {}
        for part in inner.split(","):
            if ":" not in part:
                raise TypedGraphError(f"Invalid map literal: {raw}")
            key, value = part.split(":", 1)
            out[key.strip()] = _parse_value(value.strip())
        return out
    if raw.startswith("0x") or raw.startswith("0X"):
        try:
            return int(raw, 16)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_kv_tokens(tokens: List[str], *, line_no: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise TypedGraphError(f"line {line_no}: invalid token (expected key=value): {tok}")
        key, value = tok.split("=", 1)
        args[key] = _parse_value(value)
    return args


def _parse_mem_space(value: str, *, line_no: int) -> MemSpace:
    lowered = value.lower()
    if lowered == "gmem":
        return MemSpace.GMEM
    if lowered == "smem":
        return MemSpace.SMEM
    if lowered == "tmem":
        return MemSpace.TMEM
    if lowered == "rmem":
        return MemSpace.RMEM
    raise TypedGraphError(f"line {line_no}: unknown mem space '{value}'")


@dataclass
class _IfCtx:
    node: Node
    parent: List[Node]
    branch: str  # then|else


@dataclass
class _ForCtx:
    node: Node
    parent: List[Node]


def _current_list(g: Graph, section: str, stack: List[Any]) -> List[Node]:
    if stack:
        top = stack[-1]
        if isinstance(top, _IfCtx):
            if top.branch == "then":
                then_node = next((c for c in top.node.children if c.kind == "Then"), None)
                if then_node is None:
                    raise TypedGraphError("internal error: missing Then node")
                return then_node.children
            else:
                else_node = next((c for c in top.node.children if c.kind == "Else"), None)
                if else_node is None:
                    raise TypedGraphError("internal error: missing Else node")
                return else_node.children
        if isinstance(top, _ForCtx):
            return top.node.children
        raise TypedGraphError("internal error: unknown control-flow stack item")
    return g.sections[section]


def _parse_op(op_name: str, op_args: Dict[str, Any], loc: SourceLoc) -> OpNode:
    _infer_op_metadata(op_name, op_args)
    op_cls = select_op(op_name, op_args)
    if op_cls is not None:
        node = op_cls(op=op_name, args=op_args, loc=loc)
    else:
        node = OpNode(op=op_name, args=op_args, loc=loc)
    node.meta["base"] = _canonical_op_name(op_name)
    return node


def load_typed_graph(path: Path) -> Graph:
    g = Graph()
    g.meta["typed_graph"] = True
    g.meta["strict_protocol"] = True
    lines = path.read_text().splitlines()
    section: Optional[str] = None
    stack: List[Any] = []

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("# section:"):
            section_name = stripped.split(":", 1)[1].strip()
            if section_name not in g.sections:
                raise TypedGraphError(f"line {line_no}: unknown section '{section_name}'")
            section = section_name
            continue
        if stripped.startswith("#"):
            continue

        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError as exc:
            raise TypedGraphError(f"line {line_no}: {exc}") from exc
        if not tokens:
            continue

        directive = tokens[0]
        body = tokens[1:]
        loc = SourceLoc(filename=str(path), line=line_no, column=1)

        if directive == "buffer":
            args = _parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            space = _parse_mem_space(str(args.pop("space", "tmem")), line_no=line_no)
            dtype = str(args.pop("dtype", "opaque"))
            shape_val = args.pop("shape", [0])
            if isinstance(shape_val, int):
                shape = (shape_val,)
            elif isinstance(shape_val, list):
                shape = tuple(shape_val)
            else:
                shape = (shape_val,)
            g.add_buffer(name, space, shape, dtype)
            g.buffers[name].meta.update(args)
            continue

        if directive == "barrier":
            args = _parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            scope = str(args.pop("scope", "cta"))
            g.add_barrier(name=name, scope=scope)
            g.barriers[name].meta.update(args)
            continue

        if directive == "tmap":
            args = _parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            g.add_tmap(name, args)
            continue

        if directive == "desc":
            args = _parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            buf = args.pop("buf", None)
            g.add_descriptor(name, str(buf) if buf is not None else None, args)
            continue

        if directive == "default_tmem":
            args = _parse_kv_tokens(body, line_no=line_no)
            g.set_default_tmem(str(args["name"]))
            continue

        if section is None:
            raise TypedGraphError(f"line {line_no}: section not set (use '# section:<name>')")

        target = _current_list(g, section, stack)

        if directive == "kernel_start":
            args = _parse_kv_tokens(body, line_no=line_no)
            if "name" not in args:
                raise TypedGraphError(f"line {line_no}: kernel_start requires name=")
            target.append(Node(kind="KernelStart", args=args, loc=loc))
            continue

        if directive == "kernel_end":
            target.append(Node(kind="KernelEnd", args={}, loc=loc))
            continue

        if directive == "launch":
            args = _parse_kv_tokens(body, line_no=line_no)
            target.append(Node(kind="Launch", args=args, loc=loc))
            continue

        if directive == "load_inline":
            args = _parse_kv_tokens(body, line_no=line_no)
            for list_key in ("extra_cuda_cflags", "extra_ldflags", "sections"):
                if list_key in args and isinstance(args[list_key], str):
                    args[list_key] = [x for x in str(args[list_key]).split(",") if x]
            target.append(Node(kind="LoadInline", args=args, loc=loc))
            continue

        if directive == "if":
            args = _parse_kv_tokens(body, line_no=line_no)
            if "cond" not in args:
                raise TypedGraphError(f"line {line_no}: if requires cond=")
            if_node = Node(kind="If", args={"cond": str(args["cond"])}, loc=loc)
            if_node.children.append(Node(kind="Then", children=[]))
            target.append(if_node)
            stack.append(_IfCtx(node=if_node, parent=target, branch="then"))
            continue

        if directive == "else":
            if not stack or not isinstance(stack[-1], _IfCtx):
                raise TypedGraphError(f"line {line_no}: else without if")
            if_ctx = stack[-1]
            if if_ctx.branch == "else":
                raise TypedGraphError(f"line {line_no}: duplicate else")
            if_ctx.node.children.append(Node(kind="Else", children=[]))
            if_ctx.branch = "else"
            continue

        if directive == "endif":
            if not stack or not isinstance(stack[-1], _IfCtx):
                raise TypedGraphError(f"line {line_no}: endif without if")
            stack.pop()
            continue

        if directive == "for":
            args = _parse_kv_tokens(body, line_no=line_no)
            if "var" not in args or "iters" not in args:
                raise TypedGraphError(f"line {line_no}: for requires var= and iters=")
            for_node = Node(
                kind="For",
                args={"var": str(args["var"]), "iters": str(args["iters"])},
                children=[],
                loc=loc,
            )
            target.append(for_node)
            stack.append(_ForCtx(node=for_node, parent=target))
            continue

        if directive == "endfor":
            if not stack or not isinstance(stack[-1], _ForCtx):
                raise TypedGraphError(f"line {line_no}: endfor without for")
            stack.pop()
            continue

        if directive == "op":
            if not body:
                raise TypedGraphError(f"line {line_no}: op requires operation name")
            op_name = body[0]
            op_args = _parse_kv_tokens(body[1:], line_no=line_no)
            target.append(_parse_op(op_name, op_args, loc))
            continue

        if directive == "node":
            if not body:
                raise TypedGraphError(f"line {line_no}: node requires kind")
            kind = body[0]
            args = _parse_kv_tokens(body[1:], line_no=line_no)
            target.append(Node(kind=kind, args=args, loc=loc))
            continue

        raise TypedGraphError(f"line {line_no}: unknown directive '{directive}'")

    if stack:
        raise TypedGraphError(f"{path}: unclosed control-flow block(s)")
    return g


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ",".join(f"{k}:{_format_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    text = str(value)
    if not text:
        return "''"
    if any(ch.isspace() for ch in text) or any(ch in text for ch in ['"', "'", "[", "]", "{", "}", ","]):
        return repr(text)
    return text


def _format_kv(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    return " " + " ".join(f"{k}={_format_value(v)}" for k, v in args.items())


def _iter_emit_nodes(nodes: Iterable[Node], lines: List[str]) -> None:
    for node in nodes:
        if node.kind == "Raw":
            continue
        if node.kind == "KernelStart":
            lines.append(f"kernel_start{_format_kv(node.args)}")
            continue
        if node.kind == "KernelEnd":
            lines.append("kernel_end")
            continue
        if node.kind == "Launch":
            lines.append(f"launch{_format_kv(node.args)}")
            continue
        if node.kind == "LoadInline":
            lines.append(f"load_inline{_format_kv(node.args)}")
            continue
        if node.kind == "For":
            lines.append(f"for{_format_kv(node.args)}")
            _iter_emit_nodes(node.children, lines)
            lines.append("endfor")
            continue
        if node.kind == "If":
            cond = node.args.get("cond", "")
            lines.append(f"if cond={_format_value(cond)}")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            else_node = next((c for c in node.children if c.kind == "Else"), None)
            if then_node is not None:
                _iter_emit_nodes(then_node.children, lines)
            if else_node is not None:
                lines.append("else")
                _iter_emit_nodes(else_node.children, lines)
            lines.append("endif")
            continue
        if node.kind == "Op":
            op = str(node.args.get("op", ""))
            op_args = dict(node.args.get("op_args", {}))
            lines.append(f"op {op}{_format_kv(op_args)}")
            continue
        if node.kind in {"Then", "Else"}:
            _iter_emit_nodes(node.children, lines)
            continue
        lines.append(f"node {node.kind}{_format_kv(node.args)}")


def dump_typed_graph(g: Graph, path: Path) -> None:
    lines: List[str] = ["# typed_graph:v1"]

    for name, buf in g.buffers.items():
        args: Dict[str, Any] = {
            "name": name,
            "space": buf.space.name.lower(),
            "dtype": buf.dtype,
            "shape": list(buf.shape),
        }
        args.update(buf.meta)
        lines.append(f"buffer{_format_kv(args)}")

    for name, bar in g.barriers.items():
        args = {"name": name, "scope": bar.scope}
        args.update(bar.meta)
        lines.append(f"barrier{_format_kv(args)}")

    for name, desc in g.descriptors.items():
        args = {"name": name}
        if desc.buf is not None:
            args["buf"] = desc.buf
        args.update(desc.meta)
        lines.append(f"desc{_format_kv(args)}")

    if g.default_tmem:
        lines.append(f"default_tmem name={_format_value(g.default_tmem)}")

    for section in ("device", "host", "python"):
        lines.append("")
        lines.append(f"# section:{section}")
        _iter_emit_nodes(g.sections.get(section, []), lines)

    path.write_text("\n".join(lines) + "\n")
