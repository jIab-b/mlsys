#!/usr/bin/env python3
"""Build cuda_lib files from a DSL graph with static validation for tcgen ops.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; cuda_lib/* holds
host/device/python raw code. Non-tcgen code is emitted via Raw nodes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CUDA_LIB = REPO_ROOT / "cuda_lib"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.core import Graph, MemSpace  # noqa: E402
from emit import _emit_nodes, emit_section  # noqa: E402
from parse import (  # noqa: E402
    _split_with_annotations,
    _split_python_with_load_inline,
    load_section_nodes,
)
from pretty import graph_string  # noqa: E402
from validate import validate_graph  # noqa: E402

SECTION_FILES = {
    "device": CUDA_LIB / "device.cuh",
    "host": CUDA_LIB / "host.cuh",
    "python": CUDA_LIB / "python.py",
}


def build_gemm1_graph() -> Graph:
    g = Graph()
    missing = [name for name, path in SECTION_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cuda_lib files for gemm1: {missing}")

    load_section_nodes("device", SECTION_FILES["device"], g)
    if not g.buffers:
        g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")
    load_section_nodes("host", SECTION_FILES["host"], g)
    python_text = SECTION_FILES["python"].read_text()
    pre_py, post_py = _split_python_with_load_inline(python_text)
    for node in _split_with_annotations(pre_py, SECTION_FILES["python"], g):
        g.sections["python"].append(node)
    g.add_load_inline(
        "python",
        name="gemm_all",
        cuda_src_var="CUDA_SRC",
        cpp_sources="",
        verbose=False,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        extra_ldflags=["-lcuda"],
    )
    for node in _split_with_annotations(post_py, SECTION_FILES["python"], g):
        g.sections["python"].append(node)
    return g


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile graph into cuda_lib/*")
    parser.add_argument("--dump-graph", action="store_true", help="Print graph structure")
    parser.add_argument("--no-emit", action="store_true", help="Skip writing cuda_lib files")
    args = parser.parse_args()

    g = build_gemm1_graph()
    validate_graph(g)
    if args.dump_graph:
        print(graph_string(g))
    if not args.no_emit:
        for section, path in SECTION_FILES.items():
            emit_section(
                g.sections.get(section, []),
                path,
                emit_load_inline=(section != "python"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
