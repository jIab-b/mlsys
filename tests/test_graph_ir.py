#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def _iter_nodes(nodes: Iterable[object]) -> Iterable[object]:
    for node in nodes:
        yield node
        children = getattr(node, "children", None) or []
        yield from _iter_nodes(children)


def _run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "graph"))

    from state_machine import validate_graph  # noqa: WPS433
    from syntax import load_typed_graph  # noqa: WPS433

    graph_dir = repo_root / "graph" / "kernel_graphs"
    checks = [
        ("gemm1.graph", {"kernel_v4", "kernel_v3b"}),
        ("grouped_gemm.graph", {"grouped_gemm_kernel"}),
    ]

    for file_name, expected_kernels in checks:
        graph_path = graph_dir / file_name
        graph = load_typed_graph(graph_path)
        validate_graph(graph)

        found_kernels = {
            str(node.args.get("name"))
            for node in _iter_nodes(graph.sections["device"])
            if getattr(node, "kind", "") == "KernelStart"
        }
        if found_kernels != expected_kernels:
            raise AssertionError(f"{file_name}: kernels {found_kernels} != expected {expected_kernels}")

        raw_nodes = [
            node for section in graph.sections.values()
            for node in _iter_nodes(section)
            if getattr(node, "kind", "") == "Raw"
        ]
        if raw_nodes:
            raise AssertionError(f"{file_name}: expected zero Raw nodes, found {len(raw_nodes)}")

    grouped = load_typed_graph(graph_dir / "grouped_gemm.graph")
    grouped_ops = [
        (str(node.args.get("op", "")), dict(node.args.get("op_args", {})))
        for node in _iter_nodes(grouped.sections["device"])
        if getattr(node, "kind", "") == "Op"
    ]
    has_group2_marker = any(name == "cta_group_set" and args.get("value") == 2 for name, args in grouped_ops)
    has_tma_store = any(name in {"tma_store_out", "tma_1d_smem2gmem", "tma_2d_smem2gmem", "tma_3d_smem2gmem"} for name, _ in grouped_ops)
    if not has_group2_marker:
        raise AssertionError("grouped_gemm.graph: missing explicit cta_group_set value=2")
    if not has_tma_store:
        raise AssertionError("grouped_gemm.graph: missing explicit output store op")


if __name__ == "__main__":
    _run()
