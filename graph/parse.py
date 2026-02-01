from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constants import LOAD_INLINE_MARKER
from graph.core import Graph, MemSpace, Node, SourceLoc
from graph.nodes.op import OpNode
from ops.tcgen05 import select_tcgen_op
from specs import GRAPH_TMEM_MAX_COLS, _canonical_op_name, _infer_op_metadata

_ANNOT_RE = re.compile(r"^\s*(?://|#)\s*@(?P<kind>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<body>.*)$")
_CHUNK_ASSIGN_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?P<prefix>[rR])?(?P<quote>\"\"\"|''')\s*$")


def _parse_value(raw: str) -> Any:
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
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
                tcgen_cls = select_tcgen_op(op_name, op_args)
                if tcgen_cls is not None:
                    op_node = tcgen_cls(op=op_name, args=op_args, loc=loc)
                else:
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
        else:
            # Unknown annotation: keep it in Raw, but do not validate
            pending_op = None
            pass

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
