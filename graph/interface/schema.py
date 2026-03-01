"""Agent-facing schema: JSON/dict -> Graph hydration.

AI agents produce structured JSON describing kernels. This module
converts that into a Graph the state machine can validate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from graph.ir import (
    Graph,
    MemSpace,
    Node,
    OpNode,
    RawNode,
)

_MEMSPACE_MAP = {
    "gmem": MemSpace.GMEM,
    "smem": MemSpace.SMEM,
    "tmem": MemSpace.TMEM,
    "rmem": MemSpace.RMEM,
}


def _parse_memspace(s: str) -> MemSpace:
    ms = _MEMSPACE_MAP.get(s.lower())
    if ms is None:
        raise ValueError(f"Unknown memory space: {s}")
    return ms


def _build_node(d: Dict[str, Any]) -> Node:
    """Recursively build a Node from a dict."""
    kind = d.get("kind", "")

    if kind == "Raw":
        return RawNode(code=d.get("code", ""), meta=d.get("meta"))

    if kind == "Op":
        return OpNode(op=d["op"], args=d.get("op_args"))

    children = [_build_node(c) for c in d.get("children", [])]
    return Node(kind=kind, args=d.get("args", {}), children=children, meta=d.get("meta", {}))


def graph_from_dict(data: Dict[str, Any]) -> Graph:
    """Hydrate a Graph from a JSON-compatible dict.

    Expected schema:
    {
        "buffers": [{"name": str, "space": str, "shape": list, "dtype": str, "meta": dict}],
        "barriers": [{"name": str, "scope": str}],
        "descriptors": [{"name": str, "buf": str|null, "meta": dict}],
        "tmaps": [{"name": str, ...metadata...}],
        "nodes": [node_dict, ...],
        "meta": dict
    }
    """
    g = Graph()

    for buf in data.get("buffers", []):
        g.add_buffer(
            name=buf["name"],
            space=_parse_memspace(buf["space"]),
            shape=tuple(buf.get("shape", ())),
            dtype=buf.get("dtype", "f16"),
        )
        if "meta" in buf:
            g.buffers[buf["name"]].meta.update(buf["meta"])

    for bar in data.get("barriers", []):
        g.add_barrier(name=bar["name"], scope=bar.get("scope", "cta"))

    for desc in data.get("descriptors", []):
        g.add_descriptor(
            name=desc["name"],
            buf=desc.get("buf"),
            meta=desc.get("meta", {}),
        )

    for tmap in data.get("tmaps", []):
        name = tmap["name"]
        meta = {k: v for k, v in tmap.items() if k != "name"}
        g.add_tmap(name, meta)

    for node_data in data.get("nodes", []):
        g.add_node(_build_node(node_data))

    if "meta" in data:
        g.meta.update(data["meta"])

    return g
