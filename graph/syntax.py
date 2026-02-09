from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from constants import LOAD_INLINE_MARKER
from ir import Graph, MemSpace, Node, OpNode, SourceLoc
from ptx_ops.utils.spec import GRAPH_TMEM_MAX_COLS, _canonical_op_name, _infer_op_metadata


# -------------------------
# Shared lexical helpers
# -------------------------

_ANNOT_RE = re.compile(r"^\s*(?://|#)\s*@(?P<kind>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<body>.*)$")
_CHUNK_ASSIGN_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?P<prefix>[rR])?(?P<quote>\"\"\"|''')\s*$")


def _parse_value(raw: str) -> Any:
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    mul = re.fullmatch(r"(-?\d+)\s*\*\s*(-?\d+)", raw)
    if mul:
        return int(mul.group(1)) * int(mul.group(2))
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
                return raw
            key, value = part.split(":", 1)
            out[key.strip()] = _parse_value(value.strip())
        return out
    if re.fullmatch(r"0x[0-9a-fA-F]+", raw):
        return int(raw, 16)
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def _parse_kv_tokens(tokens: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Invalid annotation token (expected key=value): {tok}")
        key, value = tok.split("=", 1)
        args[key] = _parse_value(value)
    return args


# -------------------------
# Legacy annotation parser
# -------------------------


def _parse_chunked_source(text: str, filename: Path) -> Optional[List[Tuple[str, str, int]]]:
    """Parse chunked triple-quoted strings into (name, code, start_line)."""
    lines = text.splitlines()
    chunks: List[Tuple[str, str, int]] = []
    i = 0
    while i < len(lines):
        m = _ANNOT_RE.match(lines[i])
        if not m or m.group("kind") != "chunk":
            i += 1
            continue
        body = m.group("body").strip()
        args = _parse_kv_tokens(shlex.split(body)) if body else {}
        name = str(args.get("name") or args.get("id") or f"chunk_{len(chunks)}")
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i >= len(lines):
            raise ValueError(f"{filename}:{i}: @chunk '{name}' missing triple-quoted string")
        m_assign = _CHUNK_ASSIGN_RE.match(lines[i])
        if not m_assign:
            raise ValueError(f"{filename}:{i+1}: expected triple-quoted string after @chunk '{name}'")
        quote = m_assign.group("quote")
        i += 1
        start_line = i + 1
        content_lines: List[str] = []
        while i < len(lines):
            if lines[i].strip() == quote:
                break
            content_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError(f"{filename}:{start_line}: unterminated triple-quoted string for chunk '{name}'")
        chunk_text = "\n".join(content_lines)
        if chunk_text and not chunk_text.endswith("\n"):
            chunk_text += "\n"
        chunks.append((name, chunk_text, start_line))
        i += 1
    return chunks or None


def _register_buffer(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("@buffer requires name=")
    space = str(args.get("space", "tmem")).lower()
    if space == "tmem":
        mem = MemSpace.TMEM
    elif space == "smem":
        mem = MemSpace.SMEM
    elif space == "gmem":
        mem = MemSpace.GMEM
    else:
        mem = MemSpace.RMEM
    cols = args.get("cols")
    shape = (cols,) if cols is not None else (0,)
    g.add_buffer(str(name), mem, shape, str(args.get("dtype", "opaque")))
    g.buffers[str(name)].meta.update(args)
    if mem == MemSpace.TMEM and isinstance(cols, int) and cols > GRAPH_TMEM_MAX_COLS:
        raise ValueError(f"tmem buffer '{name}' cols {cols} > {GRAPH_TMEM_MAX_COLS}")


def _register_barrier(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("@barrier requires name=")
    scope = str(args.get("scope", "cta"))
    g.add_barrier(str(name), scope=scope)
    g.barriers[str(name)].meta.update(args)


def _register_tmap(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("cute_tmap requires name=")
    g.add_tmap(str(name), args)


def _register_descriptor(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("@desc requires name=")
    buf = args.get("buf")
    g.add_descriptor(str(name), str(buf) if buf is not None else None, args)


def _split_with_annotations(
    text: str,
    filename: Path,
    g: Graph,
    *,
    line_offset: int = 0,
    initial_chunk: Optional[str] = None,
    allow_chunk: bool = True,
) -> List[Node]:
    nodes: List[Node] = []
    raw_lines: List[str] = []
    events: List[Node] = []
    loop_stack: List[Dict[str, Any]] = []
    chunk_name: Optional[str] = initial_chunk
    pending_op: Optional[OpNode] = None

    def flush_chunk() -> None:
        nonlocal raw_lines, events, chunk_name, pending_op
        if raw_lines:
            meta = {"chunk": chunk_name} if chunk_name else {}
            nodes.append(Node(kind="Raw", args={"code": "\n".join(raw_lines)}, meta=meta))
        if events:
            nodes.extend(events)
        raw_lines = []
        events = []
        chunk_name = None
        pending_op = None

    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        m = _ANNOT_RE.match(line)
        if not m:
            raw_lines.append(line)
            pending_op = None
            continue

        kind = m.group("kind")
        body = m.group("body").strip()
        loc = SourceLoc(filename=str(filename), line=idx + line_offset, column=1)

        if kind == "chunk":
            if allow_chunk:
                flush_chunk()
                raw_lines.append(line)
                args = _parse_kv_tokens(shlex.split(body)) if body else {}
                name = args.get("name") or args.get("id")
                chunk_name = str(name) if name is not None else None
                pending_op = None
                continue

        # keep annotation line in raw chunk
        raw_lines.append(line)

        if kind == "op":
            tokens = shlex.split(body)
            if not tokens or (tokens and "=" in tokens[0]):
                if pending_op is None:
                    raise ValueError(f"{filename}:{idx}: @op continuation without prior @op")
                extra_args = _parse_kv_tokens(tokens) if tokens else {}
                if "when" in extra_args:
                    raise ValueError(f"{filename}:{idx}: @op continuation cannot introduce 'when'")
                pending_op.args.setdefault("op_args", {}).update(extra_args)
                _infer_op_metadata(str(pending_op.args.get("op", "")), pending_op.args.get("op_args", {}))
                pending_op = None
            else:
                op_name = tokens[0]
                op_args = _parse_kv_tokens(tokens[1:])
                if loop_stack:
                    op_args["_loops"] = [dict(entry) for entry in loop_stack]
                when_cond = op_args.pop("when", None)
                _infer_op_metadata(op_name, op_args)
                op_node = OpNode(op=op_name, args=op_args, loc=loc)
                op_node.meta["base"] = _canonical_op_name(op_name)
                if when_cond is not None:
                    if_node = Node(kind="If", args={"cond": str(when_cond)}, meta={"validate_only": "true"})
                    if_node.children.append(Node(kind="Then", children=[op_node]))
                    events.append(if_node)
                else:
                    events.append(op_node)
                pending_op = op_node
        elif kind == "buffer":
            args = _parse_kv_tokens(shlex.split(body))
            _register_buffer(g, args)
            pending_op = None
        elif kind == "barrier":
            args = _parse_kv_tokens(shlex.split(body))
            _register_barrier(g, args)
            pending_op = None
        elif kind == "desc":
            args = _parse_kv_tokens(shlex.split(body))
            _register_descriptor(g, args)
            pending_op = None
        elif kind == "loop":
            args = _parse_kv_tokens(shlex.split(body))
            if "var" not in args:
                raise ValueError(f"{filename}:{idx}: @loop requires var=")
            loop_stack.append(args)
            pending_op = None
        elif kind == "endloop":
            if not loop_stack:
                raise ValueError(f"{filename}:{idx}: @endloop without @loop")
            loop_stack.pop()
            pending_op = None
        elif kind == "kernel":
            args = _parse_kv_tokens(shlex.split(body))
            name = str(args.get("name", "kernel"))
            args["name"] = name
            events.append(Node(kind="KernelStart", args=args, loc=loc))
            pending_op = None
        elif kind == "endkernel":
            events.append(Node(kind="KernelEnd", args={}, loc=loc))
            pending_op = None
        elif kind == "launch":
            args = _parse_kv_tokens(shlex.split(body)) if body else {}
            events.append(Node(kind="Launch", args=args, loc=loc))
            pending_op = None
        elif kind == "tmap":
            args = _parse_kv_tokens(shlex.split(body)) if body else {}
            _register_tmap(g, args)
            pending_op = None
        else:
            pending_op = None

    flush_chunk()
    if loop_stack:
        raise ValueError(f"{filename}: unterminated @loop annotations")
    return nodes


def _split_python_with_load_inline(text: str) -> Tuple[str, str]:
    if LOAD_INLINE_MARKER not in text:
        raise ValueError(f"python.py missing load inline marker: {LOAD_INLINE_MARKER}")
    pre, post = text.split(LOAD_INLINE_MARKER, 1)
    return pre.rstrip() + "\n", post.lstrip()


def load_section_nodes(section: str, path: Path, g: Graph) -> None:
    text = path.read_text()
    chunked = _parse_chunked_source(text, path)
    if chunked:
        for name, chunk_text, start_line in chunked:
            for node in _split_with_annotations(
                chunk_text,
                path,
                g,
                line_offset=start_line - 1,
                initial_chunk=name,
                allow_chunk=False,
            ):
                g.sections[section].append(node)
        return
    for node in _split_with_annotations(text, path, g):
        g.sections[section].append(node)


# -------------------------
# Emitter
# -------------------------


def _emit_nodes(
    nodes: List[Node],
    indent: int = 0,
    emit_load_inline: bool = True,
    marker_state: Optional[List[bool]] = None,
) -> List[str]:
    lines: List[str] = []
    pad = " " * indent
    for node in nodes:
        if node.kind == "Block":
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            continue
        if node.kind == "If":
            if node.meta.get("validate_only") == "true":
                continue
            cond = node.args.get("cond", "/*cond*/")
            lines.append(f"{pad}if ({cond}) {{\n")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            if then_node:
                lines.extend(
                    _emit_nodes(
                        then_node.children,
                        indent + 2,
                        emit_load_inline=emit_load_inline,
                        marker_state=marker_state,
                    )
                )
            lines.append(f"{pad}}}\n")
            else_node = next((c for c in node.children if c.kind == "Else"), None)
            if else_node:
                lines.append(f"{pad}else {{\n")
                lines.extend(
                    _emit_nodes(
                        else_node.children,
                        indent + 2,
                        emit_load_inline=emit_load_inline,
                        marker_state=marker_state,
                    )
                )
                lines.append(f"{pad}}}\n")
            continue
        if node.kind == "LoadInline":
            if not emit_load_inline:
                if marker_state is not None:
                    if not marker_state[0]:
                        lines.append(f"{pad}{LOAD_INLINE_MARKER}\n")
                        marker_state[0] = True
                else:
                    lines.append(f"{pad}{LOAD_INLINE_MARKER}\n")
                continue
            name = node.args.get("name", "module")
            cuda_src_var = node.args.get("cuda_src_var", "CUDA_SRC")
            cpp_sources = node.args.get("cpp_sources", "")
            extra_cuda_cflags = node.args.get("extra_cuda_cflags", [])
            extra_ldflags = node.args.get("extra_ldflags", [])
            verbose = node.args.get("verbose", False)
            is_python_module = node.args.get("is_python_module", False)
            no_implicit_headers = node.args.get("no_implicit_headers", True)
            lines.append(f"{pad}load_inline(\n")
            lines.append(f"{pad}    {name!r},\n")
            lines.append(f"{pad}    cpp_sources={cpp_sources!r},\n")
            lines.append(f"{pad}    cuda_sources={cuda_src_var},\n")
            lines.append(f"{pad}    verbose={verbose},\n")
            lines.append(f"{pad}    is_python_module={is_python_module},\n")
            lines.append(f"{pad}    no_implicit_headers={no_implicit_headers},\n")
            lines.append(f"{pad}    extra_cuda_cflags={extra_cuda_cflags!r},\n")
            lines.append(f"{pad}    extra_ldflags={extra_ldflags!r},\n")
            lines.append(f"{pad})\n")
            continue
        if node.kind == "Raw":
            raw = node.args.get("code", "")
            if raw:
                lines.append(raw.rstrip("\n") + "\n")
            continue
        if node.kind in ("Op", "KernelStart", "KernelEnd", "Launch"):
            continue

        if node.kind == "For":
            var = node.args.get("var", "i")
            iters = node.args.get("iters", "0")
            lines.append(f"{pad}for (int {var} = 0; {var} < {iters}; ++{var}) {{\n")
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent + 2,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            lines.append(f"{pad}}}\n")
            continue
        if node.kind in ("Then", "Else"):
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            continue

        if "call" in node.args:
            lines.append(f"{pad}{node.args['call']};\n")
        else:
            args = ", ".join(node.args.values())
            lines.append(f"{pad}{node.kind}({args});\n")
    return lines


def _iter_raw_nodes(nodes: List[Node]) -> List[Node]:
    raw_nodes: List[Node] = []
    for node in nodes:
        if node.kind == "Raw":
            raw_nodes.append(node)
        if node.children:
            raw_nodes.extend(_iter_raw_nodes(node.children))
    return raw_nodes


def _emit_chunked_section(nodes: List[Node], out_path: Path, prefix: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_nodes = _iter_raw_nodes(nodes)
    parts: List[str] = []
    for idx, node in enumerate(raw_nodes):
        name = f"{prefix}_{idx:03d}"
        var = f"{prefix.upper()}_{idx:03d}"
        parts.append(f"# @chunk name={name}\n")
        parts.append(f"{var} = r\"\"\"\n")
        code = node.args.get("code", "")
        if code:
            if not code.endswith("\n"):
                code += "\n"
            parts.append(code)
        parts.append("\"\"\"\n\n")
    out_path.write_text("".join(parts))


def emit_section(
    nodes: List[Node],
    out_path: Path,
    emit_load_inline: bool = True,
) -> None:
    if out_path.name in {"device.cuh", "host.cuh"}:
        _emit_chunked_section(nodes, out_path, prefix=out_path.stem)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    marker_state = [False] if not emit_load_inline else None
    out_path.write_text(
        "".join(
            _emit_nodes(
                nodes,
                indent=0,
                emit_load_inline=emit_load_inline,
                marker_state=marker_state,
            )
        )
    )


# -------------------------
# Graph string view
# -------------------------


def _format_node(node: Node, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if node.kind == "Raw":
        code = node.args.get("code", "")
        line_count = 0 if not code else code.count("\n") + 1
        meta = "" if not node.meta else f" meta={list(node.meta.keys())}"
        lines = [f"{pad}Raw(lines={line_count}){meta}"]
    elif node.kind == "LoadInline":
        name = node.args.get("name", "module")
        lines = [f"{pad}LoadInline(name={name})"]
    elif node.kind == "Op":
        op = node.args.get("op", "")
        base = node.meta.get("base")
        op_args = node.args.get("op_args", {})
        node_type = node.__class__.__name__
        detail_bits = []
        if isinstance(op_args, dict):
            if "shape" in op_args:
                detail_bits.append(f"shape={op_args['shape']}")
            if "num" in op_args:
                detail_bits.append(f"num={op_args['num']}")
        detail = f" {' '.join(detail_bits)}" if detail_bits else ""
        type_detail = f" type={node_type}" if node_type not in {"OpNode"} else ""
        if base and base != op:
            lines = [f"{pad}Op({op} -> {base}){detail}{type_detail}"]
        else:
            lines = [f"{pad}Op({op}){detail}{type_detail}"]
    elif node.kind == "KernelStart":
        name = node.args.get("name", "")
        lines = [f"{pad}KernelStart({name})"]
    elif node.kind == "KernelEnd":
        lines = [f"{pad}KernelEnd"]
    elif node.kind == "Launch":
        args = f" {node.args}" if node.args else ""
        lines = [f"{pad}Launch{args}"]
    else:
        args = f" {node.args}" if node.args else ""
        meta = "" if not node.meta else f" meta={list(node.meta.keys())}"
        lines = [f"{pad}{node.kind}{args}{meta}"]
    for child in node.children:
        lines.extend(_format_node(child, indent + 1))
    return lines


def graph_string(g: Graph) -> str:
    lines: List[str] = []
    lines.append("Graph:")
    lines.append(f"  buffers: {list(g.buffers.keys())}")
    lines.append(f"  barriers: {list(g.barriers.keys())}")
    lines.append(f"  tmaps: {list(g.tmaps.keys())}")
    lines.append(f"  default_tmem: {g.default_tmem}")
    for section, nodes in g.sections.items():
        lines.append(f"  section:{section} nodes={len(nodes)}")
        for line in _format_node(Node(kind="Block", children=nodes), indent=2):
            lines.append(line)
    return "\n".join(lines)


# -------------------------
# Typed graph syntax
# -------------------------


class TypedGraphError(ValueError):
    """Raised for malformed typed .graph files."""


def _typed_parse_value(raw: str) -> Any:
    return _parse_value(raw)


def _typed_parse_kv_tokens(tokens: List[str], *, line_no: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise TypedGraphError(f"line {line_no}: invalid token (expected key=value): {tok}")
        key, value = tok.split("=", 1)
        args[key] = _typed_parse_value(value)
    return args


def _typed_parse_mem_space(value: str, *, line_no: int) -> MemSpace:
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
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            space = _typed_parse_mem_space(str(args.pop("space", "tmem")), line_no=line_no)
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
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            scope = str(args.pop("scope", "cta"))
            g.add_barrier(name=name, scope=scope)
            g.barriers[name].meta.update(args)
            continue

        if directive == "tmap":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            g.add_tmap(name, args)
            continue

        if directive == "desc":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            name = str(args.pop("name"))
            buf = args.pop("buf", None)
            g.add_descriptor(name, str(buf) if buf is not None else None, args)
            continue

        if directive == "default_tmem":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            g.set_default_tmem(str(args["name"]))
            continue

        if section is None:
            raise TypedGraphError(f"line {line_no}: section not set (use '# section:<name>')")

        target = _current_list(g, section, stack)

        if directive == "kernel_start":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            if "name" not in args:
                raise TypedGraphError(f"line {line_no}: kernel_start requires name=")
            target.append(Node(kind="KernelStart", args=args, loc=loc))
            continue

        if directive == "kernel_end":
            target.append(Node(kind="KernelEnd", args={}, loc=loc))
            continue

        if directive == "launch":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            target.append(Node(kind="Launch", args=args, loc=loc))
            continue

        if directive == "load_inline":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
            for list_key in ("extra_cuda_cflags", "extra_ldflags", "sections"):
                if list_key in args and isinstance(args[list_key], str):
                    args[list_key] = [x for x in str(args[list_key]).split(",") if x]
            target.append(Node(kind="LoadInline", args=args, loc=loc))
            continue

        if directive == "if":
            args = _typed_parse_kv_tokens(body, line_no=line_no)
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
            args = _typed_parse_kv_tokens(body, line_no=line_no)
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
            op_args = _typed_parse_kv_tokens(body[1:], line_no=line_no)
            target.append(_parse_op(op_name, op_args, loc))
            continue

        if directive == "node":
            if not body:
                raise TypedGraphError(f"line {line_no}: node requires kind")
            kind = body[0]
            args = _typed_parse_kv_tokens(body[1:], line_no=line_no)
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
