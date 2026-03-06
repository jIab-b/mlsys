from __future__ import annotations

from typing import Any, Dict, Optional

from graph.ir import Graph, MemSpace
from graph.ptx_ops.spec import (
    TCGEN05_CP_SHAPE_TILE,
    TCGEN05_MMA_SHAPES,
    TCGEN05_SWIZZLE_ALIGN_BYTES,
    TCGEN05_SWIZZLE_VALID,
    TCGEN_DESC_SBO_LBO_LUT,
)


def _norm_swizzle(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    v = v.replace("swizzle", "").replace("_", "").replace("-", "")
    if v in {"none", "noswizzle", "0"}:
        return "none"
    if "128" in v and ("32a" in v or "atomic" in v):
        return "128b32a"
    if v in {"32b", "32"}:
        return "32b"
    if v in {"64b", "64"}:
        return "64b"
    if v in {"128b", "128"}:
        return "128b"
    return v


def _lookup_desc_lut(
    op_base: Optional[str],
    shape: Optional[str],
    tile: Optional[str],
    swizzle: Optional[str],
    major: Optional[str],
) -> Optional[set[tuple[int, int]]]:
    for key, allowed in TCGEN_DESC_SBO_LBO_LUT.items():
        k_op, k_shape, k_tile, k_swizzle, k_major = key
        if k_op is not None and k_op != op_base:
            continue
        if k_shape is not None and k_shape != shape:
            continue
        if k_tile is not None and k_tile != tile:
            continue
        if k_swizzle is not None and k_swizzle != swizzle:
            continue
        if k_major is not None and k_major != major:
            continue
        return allowed
    return None


def _validate_desc_ref(op_name: str, desc_name: str, g: Graph, buf_hint: Optional[str] = None) -> None:
    if desc_name not in g.descriptors:
        raise ValueError(f"{op_name}: unknown descriptor '{desc_name}'")
    desc = g.descriptors[desc_name]
    if desc.buf and desc.buf not in g.buffers:
        raise ValueError(f"{op_name}: descriptor '{desc_name}' references unknown buffer '{desc.buf}'")
    if buf_hint and desc.buf and buf_hint != desc.buf:
        raise ValueError(f"{op_name}: descriptor '{desc_name}' buffer {desc.buf} != {buf_hint}")

    buf_name = desc.buf or buf_hint
    if buf_name:
        if buf_name not in g.buffers:
            raise ValueError(f"{op_name}: descriptor '{desc_name}' references unknown buffer '{buf_name}'")
        buf = g.buffers[buf_name]
        if buf.space != MemSpace.SMEM:
            raise ValueError(f"{op_name}: descriptor '{desc_name}' buffer '{buf_name}' not in smem")

        buf_swizzle = _norm_swizzle(buf.meta.get("swizzle"))
        desc_swizzle = _norm_swizzle(desc.meta.get("swizzle"))
        if desc_swizzle and desc_swizzle not in TCGEN05_SWIZZLE_VALID:
            raise ValueError(f"{op_name}: descriptor '{desc_name}' swizzle {desc_swizzle} invalid")
        if buf_swizzle and buf_swizzle not in TCGEN05_SWIZZLE_VALID:
            raise ValueError(f"{op_name}: buffer '{buf_name}' swizzle {buf_swizzle} invalid")
        if desc_swizzle and buf_swizzle and desc_swizzle != buf_swizzle:
            raise ValueError(
                f"{op_name}: descriptor '{desc_name}' swizzle {desc_swizzle} != buffer '{buf_name}' swizzle {buf_swizzle}"
            )

        buf_major = buf.meta.get("major")
        desc_major = desc.meta.get("major")
        if buf_major and desc_major and str(buf_major).upper() != str(desc_major).upper():
            raise ValueError(
                f"{op_name}: descriptor '{desc_name}' major {desc_major} != buffer '{buf_name}' major {buf_major}"
            )

        effective_swizzle = desc_swizzle or buf_swizzle
        if effective_swizzle in TCGEN05_SWIZZLE_ALIGN_BYTES:
            sbo = desc.meta.get("sbo")
            if isinstance(sbo, int):
                align = TCGEN05_SWIZZLE_ALIGN_BYTES[effective_swizzle]
                base_offset = desc.meta.get("base_offset")
                if sbo % align == 0:
                    if base_offset not in (None, 0):
                        raise ValueError(
                            f"{op_name}: descriptor '{desc_name}' base_offset {base_offset} must be 0 when sbo {sbo} is {align}-byte aligned"
                        )
                else:
                    if base_offset is None:
                        raise ValueError(
                            f"{op_name}: descriptor '{desc_name}' sbo {sbo} not {align}-byte aligned for swizzle {effective_swizzle}; base_offset required"
                        )
                    if not isinstance(base_offset, int):
                        raise ValueError(
                            f"{op_name}: descriptor '{desc_name}' base_offset must be integer when sbo {sbo} is misaligned"
                        )
                    pattern_start = sbo - (sbo % align)
                    expected = (pattern_start >> 7) & 0x7
                    if base_offset != expected:
                        raise ValueError(
                            f"{op_name}: descriptor '{desc_name}' base_offset {base_offset} != expected {expected} for sbo {sbo}"
                        )

    for key in ("sbo", "lbo", "stride", "leading"):
        if key not in desc.meta:
            continue
        val = desc.meta.get(key)
        if isinstance(val, int):
            if key == "lbo" and val in (0, 1):
                continue
            if val % 16 != 0:
                raise ValueError(f"{op_name}: descriptor '{desc_name}' {key}={val} not 16-byte aligned")
            if (val >> 4) > 0x3FFFF:
                raise ValueError(f"{op_name}: descriptor '{desc_name}' {key}={val} out of 14-bit range")


def validate_tcgen_descriptor_layout(canonical: str, op_name: str, op_args: Dict[str, Any], g: Graph) -> None:
    if canonical == "tcgen05_cp":
        shape = op_args.get("shape")
        tile = op_args.get("tile")
        if isinstance(shape, str) and (shape, tile) not in TCGEN05_CP_SHAPE_TILE:
            raise ValueError(f"{op_name}: shape/tile {(shape, tile)} not in {sorted(TCGEN05_CP_SHAPE_TILE)}")

        desc_name = op_args.get("desc")
        if desc_name is not None:
            desc = g.descriptors.get(str(desc_name))
            swizzle = _norm_swizzle(desc.meta.get("swizzle")) if desc else None
            major = str(desc.meta.get("major")).upper() if desc and desc.meta.get("major") else None
            allowed = _lookup_desc_lut("tcgen05_cp", op_args.get("shape"), op_args.get("tile"), swizzle, major)
            _validate_desc_ref(op_name, str(desc_name), g, buf_hint=op_args.get("smem_buf"))
            if allowed is not None:
                if not isinstance(desc.meta.get("sbo"), int) or not isinstance(desc.meta.get("lbo"), int):
                    raise ValueError(f"{op_name}: descriptor '{desc_name}' requires numeric sbo/lbo for LUT check")
                pair = (int(desc.meta["sbo"]), int(desc.meta["lbo"]))
                if pair not in allowed:
                    raise ValueError(f"{op_name}: descriptor '{desc_name}' sbo/lbo {pair} not in LUT {sorted(allowed)}")
        elif "smem_buf" in op_args:
            buf_name = op_args.get("smem_buf")
            if buf_name not in g.buffers:
                raise ValueError(f"{op_name}: unknown smem_buf '{buf_name}'")

    if canonical == "tcgen05_mma":
        shape = op_args.get("shape")
        if isinstance(shape, str) and shape not in TCGEN05_MMA_SHAPES:
            raise ValueError(f"{op_name}: shape {shape} not in {sorted(TCGEN05_MMA_SHAPES)}")

        for desc_key in ("desc_a", "desc_b"):
            desc_name = op_args.get(desc_key)
            if desc_name is None:
                continue
            desc = g.descriptors.get(str(desc_name))
            swizzle = _norm_swizzle(desc.meta.get("swizzle")) if desc else None
            major = str(desc.meta.get("major")).upper() if desc and desc.meta.get("major") else None
            allowed = _lookup_desc_lut("tcgen05_mma", op_args.get("shape"), op_args.get("tile"), swizzle, major)
            _validate_desc_ref(op_name, str(desc_name), g)
            if allowed is not None:
                if not isinstance(desc.meta.get("sbo"), int) or not isinstance(desc.meta.get("lbo"), int):
                    raise ValueError(f"{op_name}: descriptor '{desc_name}' requires numeric sbo/lbo for LUT check")
                pair = (int(desc.meta["sbo"]), int(desc.meta["lbo"]))
                if pair not in allowed:
                    raise ValueError(f"{op_name}: descriptor '{desc_name}' sbo/lbo {pair} not in LUT {sorted(allowed)}")
