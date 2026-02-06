from __future__ import annotations

from pathlib import Path

from compiler import _write_sub_test
from ptx_ops.utils.ir import Graph

from .base import BackendSpec


SPEC = BackendSpec(
    name="ptx_inline",
    implemented=True,
    description="Current emitter: inline referenced ptx_lib/*.cuh into CUDA_SRC and write submission Python.",
)


def emit(graph: Graph, out_path: Path) -> None:
    _write_sub_test(graph, out_path)
