from __future__ import annotations

from typing import List

from graph.core import Node


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
        detail_bits = []
        if isinstance(op_args, dict):
            if "shape" in op_args:
                detail_bits.append(f"shape={op_args['shape']}")
            if "num" in op_args:
                detail_bits.append(f"num={op_args['num']}")
        detail = f" {' '.join(detail_bits)}" if detail_bits else ""
        if base and base != op:
            lines = [f"{pad}Op({op} -> {base}){detail}"]
        else:
            lines = [f"{pad}Op({op}){detail}"]
    elif node.kind == "KernelStart":
        name = node.args.get("name", "")
        lines = [f"{pad}KernelStart({name})"]
    elif node.kind == "KernelEnd":
        lines = [f"{pad}KernelEnd"]
    else:
        args = f" {node.args}" if node.args else ""
        meta = "" if not node.meta else f" meta={list(node.meta.keys())}"
        lines = [f"{pad}{node.kind}{args}{meta}"]
    for child in node.children:
        lines.extend(_format_node(child, indent + 1))
    return lines


def graph_string(g) -> str:
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
