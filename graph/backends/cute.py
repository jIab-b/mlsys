from __future__ import annotations

from pathlib import Path

from ptx_ops.utils.ir import Graph

from .base import BackendSpec


SPEC = BackendSpec(
    name="cute",
    implemented=False,
    description="Scaffold only: future CuTe/CUTLASS DSL lowering backend.",
)


def emit(graph: Graph, out_path: Path) -> None:
    raise NotImplementedError(
        "cute backend scaffold exists but lowering is not implemented yet."
    )
