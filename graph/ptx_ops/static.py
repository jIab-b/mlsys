"""Static per-op validation. No graph walking, no state tracking.

Takes raw op name + args dict, checks against PTX ISA constraints.
"""
from __future__ import annotations

from typing import Any, Dict

from graph.ptx_ops.spec import (
    GRAPH_TMEM_MAX_COLS,
    OP_ARG_SPECS,
    PTX_TCGEN05_CP_SHAPE_TILE,
    PTX_TCGEN05_MMA_SHAPES,
    PTX_TCGEN05_NO_TRANSPOSE_KINDS,
    TCGEN05_IDESC_LEGAL,
    TCGEN05_LD_SHAPES,
    TCGEN05_NUM_VALUES,
    TCGEN05_SMEM_DESC_LEGAL,
    TMA_INTERLEAVE_SET,
    TMA_SWIZZLE_SET,
    _canonical_op_name,
)


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


def _bits_u(value: int, lo: int, hi: int) -> int:
    return (value >> lo) & ((1 << (hi - lo + 1)) - 1)


def _check_range(op_name: str, field_name: str, val: Any, lo: int, hi: int) -> None:
    if not isinstance(val, int):
        raise ValueError(f"{op_name}: {field_name} must be integer")
    if val < lo or val > hi:
        raise ValueError(f"{op_name}: {field_name} must be in [{lo}, {hi}], got {val}")


def _validate_smem_desc(op_name: str, arg_name: str, desc: Any) -> None:
    if desc is None:
        return
    legal = TCGEN05_SMEM_DESC_LEGAL
    ranges = legal["ranges"]
    if isinstance(desc, int):
        lo, hi = ranges["raw"]
        _check_range(op_name, arg_name, desc, lo, hi)
        fixed = _bits_u(desc, 46, 48)
        fixed_lo, fixed_hi = ranges["fixed_46_48"]
        if fixed < fixed_lo or fixed > fixed_hi:
            raise ValueError(f"{op_name}: {arg_name} bits[46:48] must be {fixed_lo}")
        sw = _bits_u(desc, 61, 63)
        if sw in legal["invalid_swizzle_code"] or sw not in legal["valid_swizzle_code"]:
            raise ValueError(f"{op_name}: {arg_name} has invalid swizzle_code {sw}")
        return
    if not isinstance(desc, dict):
        raise ValueError(f"{op_name}: {arg_name} must be int or dict")

    unknown = set(desc.keys()) - legal["keys"]
    if unknown:
        raise ValueError(f"{op_name}: {arg_name} has unknown fields {sorted(unknown)}")

    for field in ("start_enc", "ld_enc", "sd_enc", "base_offset", "ld_mode", "fixed_46_48", "raw"):
        val = desc.get(field)
        if val is None:
            continue
        lo, hi = ranges[field]
        _check_range(op_name, f"{arg_name}.{field}", val, lo, hi)

    swizzle = desc.get("swizzle_code")
    if swizzle is not None:
        lo, hi = ranges["swizzle_code"]
        _check_range(op_name, f"{arg_name}.swizzle_code", swizzle, lo, hi)
        if swizzle in legal["invalid_swizzle_code"] or swizzle not in legal["valid_swizzle_code"]:
            raise ValueError(f"{op_name}: {arg_name}.swizzle_code {swizzle} is invalid")


def _validate_idesc(op_name: str, idesc: Any) -> None:
    if idesc is None:
        return
    legal = TCGEN05_IDESC_LEGAL
    ranges = legal["ranges"]
    if isinstance(idesc, int):
        lo, hi = ranges["raw"]
        _check_range(op_name, "idesc", idesc, lo, hi)
        return
    if not isinstance(idesc, dict):
        raise ValueError(f"{op_name}: idesc must be int or dict")

    unknown = set(idesc.keys()) - legal["keys"]
    if unknown:
        raise ValueError(f"{op_name}: idesc has unknown fields {sorted(unknown)}")

    for field, val in idesc.items():
        if field == "raw":
            continue
        lo, hi = ranges[field]
        _check_range(op_name, f"idesc.{field}", val, lo, hi)

    raw = idesc.get("raw")
    if raw is not None:
        lo, hi = ranges["raw"]
        _check_range(op_name, "idesc.raw", raw, lo, hi)


def validate_args(op_name: str, args: Dict[str, Any], loc_str: str = "") -> None:
    """Check required args are present per OP_ARG_SPECS."""
    spec = OP_ARG_SPECS.get(op_name)
    if not spec:
        return
    required = spec.get("required", set())
    missing = [k for k in required if k not in args]
    if missing:
        raise ValueError(f"{loc_str}{op_name}: missing args {missing}")


def validate_ptx_op(op_name: str, op_args: Dict[str, Any]) -> None:
    """Validate a single op's arguments against PTX ISA constraints."""
    canonical = _canonical_op_name(op_name)

    if canonical.startswith("tcgen05_"):
        _validate_cta_group(op_name, op_args)

    if canonical == "tcgen05_alloc":
        cols = op_args.get("cols")
        if isinstance(cols, int):
            if cols < 32 or cols > GRAPH_TMEM_MAX_COLS:
                raise ValueError(f"{op_name}: cols {cols} out of range [32, {GRAPH_TMEM_MAX_COLS}]")
            if cols & (cols - 1) != 0:
                raise ValueError(f"{op_name}: cols {cols} must be power of 2")

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
        _validate_smem_desc(op_name, "smem_desc", op_args.get("smem_desc"))

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
        _validate_idesc(op_name, op_args.get("idesc"))
        _validate_smem_desc(op_name, "smem_desc_a", op_args.get("smem_desc_a"))
        _validate_smem_desc(op_name, "smem_desc_b", op_args.get("smem_desc_b"))

    if canonical in ("tcgen05_ld", "tcgen05_st"):
        shape = op_args.get("shape")
        num = op_args.get("num")
        if isinstance(shape, str) and shape not in TCGEN05_LD_SHAPES:
            raise ValueError(f"{op_name}: shape {shape} not in {sorted(TCGEN05_LD_SHAPES)}")
        if isinstance(num, int) and num not in TCGEN05_NUM_VALUES:
            raise ValueError(f"{op_name}: num {num} not in {sorted(TCGEN05_NUM_VALUES)}")
        if "warp_id" not in op_args or "lane_id" not in op_args:
            raise ValueError(f"{op_name}: warp_id and lane_id metadata are required")
        lane_val = op_args.get("lane_id")
        if isinstance(lane_val, str) and lane_val.lower() in {"elect", "leader"}:
            raise ValueError(f"{op_name}: lane_id=elect/leader invalid for {canonical}")

    if canonical in ("tma_load", "tma_load_mcast", "tma_store"):
        _validate_cta_group(op_name, op_args)
        tmap_swizzle = op_args.get("tmap_swizzle") or op_args.get("swizzle")
        tmap_interleave = op_args.get("tmap_interleave") or op_args.get("interleave")
        if tmap_swizzle is not None and str(tmap_swizzle).lower() not in TMA_SWIZZLE_SET:
            raise ValueError(f"{op_name}: invalid tmap swizzle {tmap_swizzle}")
        if tmap_interleave is not None and str(tmap_interleave).lower() not in TMA_INTERLEAVE_SET:
            raise ValueError(f"{op_name}: invalid tmap interleave {tmap_interleave}")
        tmap_dtype = op_args.get("tmap_dtype") or op_args.get("dtype")
        if tmap_dtype is not None and not isinstance(tmap_dtype, str):
            raise ValueError(f"{op_name}: tmap dtype must be string when provided")
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")

    if canonical in ("mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"):
        phase = op_args.get("phase")
        if isinstance(phase, int) and phase not in (0, 1):
            raise ValueError(f"{op_name}: mbarrier phase must be 0 or 1, got {phase}")

    if canonical == "mbarrier_init":
        count = op_args.get("count")
        if isinstance(count, int) and count <= 0:
            raise ValueError(f"{op_name}: count must be > 0")

    if canonical in ("mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta"):
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")

    if canonical == "cta_group_set":
        value = op_args.get("value")
        if not isinstance(value, int) or value not in (1, 2):
            raise ValueError(f"{op_name}: cta_group_set value must be 1 or 2")
