from __future__ import annotations

from typing import Any, Dict, Optional

from graph.ir import Graph
from graph.ptx_ops.spec import (
    CTA_MASK_BITS,
    TMA_DTYPE_ELEMENT_SIZE_BYTES,
    TMA_DTYPE_SWIZZLE_ALLOWED,
    TMA_INTERLEAVE_SET,
    TMA_SWIZZLE_SET,
)

from .state import ValidationState


def _norm_tma_dtype(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    v = v.replace("cutensormapdatatype::", "")
    v = v.replace("cu_tensor_map_data_type::", "")
    v = v.replace("cu_tensor_map_data_type_", "")
    return v


def _norm_tma_swizzle(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    v = v.replace("cutensormapswizzle::", "")
    v = v.replace("cu_tensor_map_swizzle::", "")
    v = v.replace("cu_tensor_map_swizzle_", "")
    v = v.replace("swizzle_", "")
    return v


def _norm_tma_interleave(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    v = v.replace("cutensormapinterleave::", "")
    v = v.replace("cu_tensor_map_interleave::", "")
    v = v.replace("cu_tensor_map_interleave_", "")
    return v


def _collect_indexed(meta: Dict[str, Any], prefix: str) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for key, value in meta.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix.isdigit():
            out[int(suffix)] = value
    return out


def validate_tmap_meta(op_name: str, tmap_meta: Dict[str, Any]) -> None:
    dtype = _norm_tma_dtype(tmap_meta.get("dtype") or tmap_meta.get("data_type"))
    swizzle = _norm_tma_swizzle(tmap_meta.get("swizzle"))
    interleave = _norm_tma_interleave(tmap_meta.get("interleave"))

    rank = tmap_meta.get("rank")
    if isinstance(rank, int):
        if rank <= 0 or rank > 5:
            raise ValueError(f"{op_name}: tmap rank {rank} out of range [1, 5]")
        if interleave is not None and interleave != "none" and rank < 3:
            raise ValueError(f"{op_name}: tmap rank {rank} must be >= 3 when interleave is {interleave}")

    if dtype is None:
        raise ValueError(f"{op_name}: tmap dtype missing")
    if swizzle is None:
        raise ValueError(f"{op_name}: tmap swizzle missing")
    if interleave is None:
        raise ValueError(f"{op_name}: tmap interleave missing")

    if swizzle not in TMA_SWIZZLE_SET:
        raise ValueError(f"{op_name}: tmap swizzle '{swizzle}' invalid")
    if interleave not in TMA_INTERLEAVE_SET:
        raise ValueError(f"{op_name}: tmap interleave '{interleave}' invalid")
    if interleave == "32b" and swizzle != "32b":
        raise ValueError(f"{op_name}: tmap interleave 32b requires swizzle 32b")
    if dtype in TMA_DTYPE_SWIZZLE_ALLOWED and swizzle not in TMA_DTYPE_SWIZZLE_ALLOWED[dtype]:
        raise ValueError(f"{op_name}: tmap dtype {dtype} does not support swizzle {swizzle}")
    if dtype == "16u6_align16b" and interleave != "none":
        raise ValueError(f"{op_name}: tmap dtype {dtype} requires interleave none")

    elem_size = TMA_DTYPE_ELEMENT_SIZE_BYTES.get(dtype)

    global_dims = _collect_indexed(tmap_meta, "global_dim")
    for dim, val in global_dims.items():
        if isinstance(val, int):
            if val <= 0:
                raise ValueError(f"{op_name}: tmap global_dim{dim} must be > 0")
            if val > (1 << 32):
                raise ValueError(f"{op_name}: tmap global_dim{dim} exceeds 2^32")
    if dtype in {"16u4_align8b", "16u4_align16b"}:
        g0 = global_dims.get(0)
        if isinstance(g0, int) and g0 % 2 != 0:
            raise ValueError(f"{op_name}: tmap global_dim0 {g0} must be multiple of 2 for {dtype}")

    global_strides = _collect_indexed(tmap_meta, "global_stride")
    for dim, val in global_strides.items():
        if isinstance(val, int):
            if val <= 0:
                raise ValueError(f"{op_name}: tmap global_stride{dim} must be > 0")
            if val > (1 << 40):
                raise ValueError(f"{op_name}: tmap global_stride{dim} exceeds 2^40")

    box_dims = _collect_indexed(tmap_meta, "box_dim")
    for dim, val in box_dims.items():
        if isinstance(val, int):
            if val <= 0 or val > 256:
                raise ValueError(f"{op_name}: tmap box_dim{dim} {val} out of range [1, 256]")

    elem_strides = _collect_indexed(tmap_meta, "elem_stride")
    for dim, val in elem_strides.items():
        if isinstance(val, int):
            if val < 1 or val > 8:
                raise ValueError(f"{op_name}: tmap elem_stride{dim} {val} out of range [1, 8]")

    if elem_size is not None:
        inner = box_dims.get(0)
        if isinstance(inner, int):
            inner_bytes = inner * elem_size
            if interleave == "none" and inner_bytes % 16 != 0:
                raise ValueError(f"{op_name}: tmap box_dim0 {inner} not 16-byte aligned for dtype {dtype}")
            if swizzle in {"32b", "64b", "128b", "128b_atom_32b", "128b_atom_32b_flip_8b", "128b_atom_64b"}:
                limit = 32.0 if swizzle == "32b" else 64.0 if swizzle == "64b" else 128.0
                if inner_bytes > limit:
                    raise ValueError(
                        f"{op_name}: tmap box_dim0 {inner} ({inner_bytes} bytes) exceeds swizzle limit {limit} bytes"
                    )
    if dtype in {"16u4_align16b", "16u6_align16b"}:
        inner = box_dims.get(0)
        if isinstance(inner, int) and inner != 128:
            raise ValueError(f"{op_name}: tmap box_dim0 {inner} must be 128 for dtype {dtype}")
    for dim, val in box_dims.items():
        stride = elem_strides.get(dim)
        if isinstance(val, int) and isinstance(stride, int):
            if val % stride != 0:
                raise ValueError(f"{op_name}: tmap box_dim{dim} {val} not divisible by elem_stride{dim} {stride}")


def validate_tma_op(canonical: str, op_name: str, op_args: Dict[str, Any], g: Graph, state: ValidationState) -> None:
    if canonical in ("tma_load", "tma_load_mcast"):
        tmap = op_args.get("tmap")
        if tmap not in g.tmaps:
            raise ValueError(f"{op_name}: unknown tmap '{tmap}'")
        validate_tmap_meta(op_name, g.tmaps[tmap])
        rank_arg = op_args.get("rank")
        tmap_rank = g.tmaps[tmap].get("rank")
        if isinstance(rank_arg, int) and isinstance(tmap_rank, int) and rank_arg != tmap_rank:
            raise ValueError(f"{op_name}: rank {rank_arg} != tmap '{tmap}' rank {tmap_rank}")

        if canonical == "tma_load_mcast":
            cta_mask = op_args.get("cta_mask")
            if isinstance(cta_mask, int):
                if cta_mask <= 0:
                    raise ValueError(f"{op_name}: cta_mask must be non-zero")
                if cta_mask >= (1 << CTA_MASK_BITS):
                    raise ValueError(f"{op_name}: cta_mask {cta_mask} exceeds {CTA_MASK_BITS} bits")
                if state.cluster_ctas is not None:
                    max_mask = (1 << state.cluster_ctas) - 1
                    if cta_mask & ~max_mask:
                        raise ValueError(f"{op_name}: cta_mask {cta_mask} outside cluster_ctas={state.cluster_ctas}")

    if canonical == "tma_store":
        tmap = op_args.get("tmap")
        if tmap not in g.tmaps:
            raise ValueError(f"{op_name}: unknown tmap '{tmap}'")
        validate_tmap_meta(op_name, g.tmaps[tmap])
        rank_arg = op_args.get("rank")
        tmap_rank = g.tmaps[tmap].get("rank")
        if isinstance(rank_arg, int) and isinstance(tmap_rank, int) and rank_arg != tmap_rank:
            raise ValueError(f"{op_name}: rank {rank_arg} != tmap '{tmap}' rank {tmap_rank}")
