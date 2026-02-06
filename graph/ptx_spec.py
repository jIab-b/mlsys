from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from graph.core import Graph, Node
from static_validator import (
    TMA_INTERLEAVE_SET,
    TMA_SWIZZLE_SET,
    _canonical_op_name,
)

# ISA references:
# - ptx_mcp/texts/ptx_docs/sections/9-7-16-2-matrix-and-data-movement-shape.md
# - ptx_mcp/texts/ptx_docs/sections/9-7-16-10-9-1-tensorcore-5th-generation-instructions-tcgen05-mma.md
# - ptx_mcp/texts/ptx_docs/sections/9-7-16-11-tensorcore-5th-generation-specialized-synchronization-operations.md
# CUDA tensor map references:
# - cuda_driver_api/texts/cuda_docs/sections/37-tensor-map-object-management.md


PTX_TCGEN05_CP_SHAPE_TILE: set[tuple[str, Optional[str]]] = {
    ("32x128b", "warpx4"),
    ("128x128b", None),
    ("128x256b", None),
    ("4x256b", None),
    ("64x128b", None),
}

PTX_TCGEN05_MMA_SHAPES: set[str] = {
    "mxf4nvf4.block16",
    "mxf4.block16",
    "f16.ss",
    "f16.ts",
    "bf16.ss",
    "bf16.ts",
    "ws.f16.ts",
}

PTX_TCGEN05_NO_TRANSPOSE_KINDS: set[str] = {"mxf4", "mxf4nvf4"}


def _iter_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_nodes(node.children)


def _shape_kind(shape: str) -> str:
    if "." not in shape:
        return shape
    return shape.split(".", 1)[0]


def _validate_cta_group(op_name: str, op_args: Dict[str, Any]) -> None:
    cta_group = op_args.get("cta_group")
    if cta_group is None:
        return
    if not isinstance(cta_group, int):
        raise ValueError(f"{op_name}: cta_group must be integer, got {type(cta_group).__name__}")
    if cta_group not in (1, 2):
        raise ValueError(f"{op_name}: cta_group must be 1 or 2, got {cta_group}")


def validate_ptx_op(op_name: str, op_args: Dict[str, Any]) -> None:
    canonical = _canonical_op_name(op_name)

    if canonical.startswith("tcgen05_"):
        _validate_cta_group(op_name, op_args)

    if canonical == "tcgen05_cp":
        shape = op_args.get("shape")
        tile = op_args.get("tile")
        if isinstance(shape, str):
            key = (shape, tile if isinstance(tile, str) else None)
            if key not in PTX_TCGEN05_CP_SHAPE_TILE:
                raise ValueError(
                    f"{op_name}: unsupported tcgen05.cp shape/tile {key}; "
                    f"allowed={sorted(PTX_TCGEN05_CP_SHAPE_TILE)}"
                )

    if canonical == "tcgen05_mma":
        shape = op_args.get("shape")
        if isinstance(shape, str) and shape not in PTX_TCGEN05_MMA_SHAPES:
            raise ValueError(
                f"{op_name}: unsupported tcgen05.mma shape {shape}; "
                f"allowed={sorted(PTX_TCGEN05_MMA_SHAPES)}"
            )
        if isinstance(shape, str):
            kind = _shape_kind(shape)
            if kind in PTX_TCGEN05_NO_TRANSPOSE_KINDS:
                ta = op_args.get("transpose_a")
                tb = op_args.get("transpose_b")
                if ta in (1, True, "1", "true", "T", "t") or tb in (1, True, "1", "true", "T", "t"):
                    raise ValueError(
                        f"{op_name}: transpose_a/transpose_b not supported for kind {kind} "
                        "(per PTX tcgen05 MMA spec)"
                    )
        idesc = op_args.get("idesc")
        if idesc is not None and isinstance(idesc, int):
            if idesc < 0 or idesc > 0xFFFFFFFF:
                raise ValueError(f"{op_name}: idesc must fit in 32-bit unsigned range")

    if canonical.startswith("tma_"):
        _validate_cta_group(op_name, op_args)
        tmap_dtype = op_args.get("tmap_dtype") or op_args.get("dtype")
        tmap_swizzle = op_args.get("tmap_swizzle") or op_args.get("swizzle")
        tmap_interleave = op_args.get("tmap_interleave") or op_args.get("interleave")
        if tmap_swizzle is not None and str(tmap_swizzle).lower() not in TMA_SWIZZLE_SET:
            raise ValueError(f"{op_name}: invalid tmap swizzle {tmap_swizzle}")
        if tmap_interleave is not None and str(tmap_interleave).lower() not in TMA_INTERLEAVE_SET:
            raise ValueError(f"{op_name}: invalid tmap interleave {tmap_interleave}")
        if tmap_dtype is not None and not isinstance(tmap_dtype, str):
            raise ValueError(f"{op_name}: tmap dtype must be string when provided")

    if canonical in ("mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"):
        phase = op_args.get("phase")
        if isinstance(phase, int) and phase not in (0, 1):
            raise ValueError(f"{op_name}: mbarrier phase must be 0 or 1, got {phase}")

    if canonical in ("tcgen05_ld", "tcgen05_st"):
        if "warp_id" not in op_args or "lane_id" not in op_args:
            raise ValueError(f"{op_name}: warp_id and lane_id metadata are required")


def validate_graph_ptx_spec(g: Graph) -> None:
    for section in ("device", "host"):
        for node in _iter_nodes(g.sections.get(section, [])):
            if node.kind != "Op":
                continue
            op_name = str(node.args.get("op", ""))
            op_args = dict(node.args.get("op_args", {}))
            validate_ptx_op(op_name, op_args)
