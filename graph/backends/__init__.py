from __future__ import annotations

from pathlib import Path
from typing import Dict

from ptx_ops.utils.ir import Graph

from .base import BackendEmitter, BackendSpec
from .cute import SPEC as CUTE_SPEC
from .cute import emit as emit_cute
from .ptx_inline import SPEC as PTX_INLINE_SPEC
from .ptx_inline import emit as emit_ptx_inline


BACKEND_SPECS: Dict[str, BackendSpec] = {
    PTX_INLINE_SPEC.name: PTX_INLINE_SPEC,
    CUTE_SPEC.name: CUTE_SPEC,
}

BACKEND_EMITTERS: Dict[str, BackendEmitter] = {
    "ptx_inline": emit_ptx_inline,
    "cute": emit_cute,
}


def backend_names() -> list[str]:
    return sorted(BACKEND_EMITTERS.keys())


def emit_with_backend(name: str, graph: Graph, out_path: Path) -> None:
    if name not in BACKEND_EMITTERS:
        known = ", ".join(backend_names())
        raise ValueError(f"Unknown backend '{name}', choose one of: {known}")
    BACKEND_EMITTERS[name](graph, out_path)
