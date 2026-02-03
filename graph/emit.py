from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from constants import LOAD_INLINE_MARKER
from graph.core import Node


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
