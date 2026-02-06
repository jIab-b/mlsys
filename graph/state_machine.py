from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graph.core import BarrierState, BufferState, Graph, MemSpace, Node, SourceLoc
from protocol_validator import validate_graph_protocol
from ptx_spec import validate_graph_ptx_spec
from static_validator import (
    BARRIER_SCOPES,
    CTA_MASK_BITS,
    GRAPH_SMEM_LIMIT_BYTES,
    GRAPH_TMEM_MAX_COLS,
    ISSUE_SCOPES,
    OP_ARG_SPECS,
    TCGEN05_CP_SHAPE_TILE,
    TCGEN05_LD_SHAPES,
    TCGEN05_MMA_SHAPES,
    TCGEN05_NUM_VALUES,
    TCGEN05_SWIZZLE_ALIGN_BYTES,
    TCGEN05_SWIZZLE_VALID,
    TCGEN_DESC_SBO_LBO_LUT,
    TMA_DTYPE_ELEMENT_SIZE_BYTES,
    TMA_DTYPE_SWIZZLE_ALLOWED,
    TMA_INTERLEAVE_SET,
    TMA_SWIZZLE_SET,
    _canonical_op_name,
    _resolve_contract,
)


@dataclass
class ValidationState:
    bar_state: Dict[str, Optional[BarrierState]]
    buf_state: Dict[str, Optional[BufferState]]
    bar_init_count: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_arrivals: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_phase: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_expected_bytes: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_completed_bytes: Dict[str, Optional[int]] = field(default_factory=dict)
    cluster_init_fenced: Optional[bool] = None
    cluster_sync_done: Optional[bool] = None
    cluster_ctas: Optional[int] = None
    pending_ld: Optional[bool] = False
    pending_st: Optional[bool] = False
    pending_tcgen_commit: Optional[bool] = False
    cta_group: Optional[int] = None
    last_alloc_cols: Optional[int] = None


def _resolve_buffer_arg(op: Node, key: str, g: Graph) -> str:
    op_args = op.args.get("op_args") if op.kind == "Op" else None
    if op_args is not None and key in op_args:
        return op_args[key]
    if key in op.args:
        return op.args[key]
    if key == "tmem" and g.default_tmem is not None:
        return g.default_tmem
    raise ValueError(f"{op.kind}: missing buffer arg '{key}'")


def _get_op_info(op: Node) -> Tuple[str, Dict[str, Any], Optional[SourceLoc]]:
    if op.kind == "Op":
        return str(op.args.get("op", "")), dict(op.args.get("op_args", {})), op.loc
    return op.kind, dict(op.args), op.loc


def _validate_args(op_name: str, args: Dict[str, Any], loc: Optional[SourceLoc]) -> None:
    spec = OP_ARG_SPECS.get(op_name)
    if not spec:
        return
    required = spec.get("required", set())
    missing = [k for k in required if k not in args]
    if missing:
        loc_str = f"{loc.filename}:{loc.line}: " if loc else ""
        raise ValueError(f"{loc_str}{op_name}: missing args {missing}")


def _validate_op(op: Node, g: Graph, state: ValidationState) -> None:
    op_name, op_args, loc = _get_op_info(op)
    if not op_name:
        raise ValueError("Op node missing name")
    canonical = _canonical_op_name(op_name)
    c = _resolve_contract(canonical)
    if c is None:
        raise ValueError(f"Unknown op: {op_name}")

    _validate_args(canonical, op_args, loc)

    if canonical in {"cute_tmap", "tmap_create"}:
        g.add_tmap(str(op_args["name"]), op_args)
        return

    if canonical == "cta_group_set":
        value = op_args.get("value")
        if not isinstance(value, int) or value not in (1, 2):
            raise ValueError(f"{op_name}: cta_group_set value must be 1 or 2")
        if state.cta_group is None:
            state.cta_group = value
        elif state.cta_group != value:
            raise ValueError(f"{op_name}: cta_group_set {value} != prior {state.cta_group}")
        return

    scope_val = op_args.get("scope")
    issue_val = op_args.get("issue") or op_args.get("issue_scope")
    if issue_val is None and isinstance(scope_val, str) and scope_val in ISSUE_SCOPES:
        issue_val = scope_val
    if issue_val is not None and issue_val != c.issue_scope:
        raise ValueError(f"{op_name}: issue_scope {issue_val} != {c.issue_scope}")

    if isinstance(scope_val, str) and scope_val in BARRIER_SCOPES and "bar" in op_args:
        bar_name = op_args["bar"]
        if bar_name in g.barriers and g.barriers[bar_name].scope != scope_val:
            raise ValueError(f"{op_name}: barrier '{bar_name}' scope {g.barriers[bar_name].scope} != {scope_val}")

    # tcgen05 cta_group consistency
    if canonical.startswith("tcgen05_"):
        cta_group = op_args.get("cta_group", 1)
        if isinstance(cta_group, int) and cta_group not in (1, 2):
            raise ValueError(f"{op_name}: invalid cta_group {cta_group}")
        if state.cta_group is None:
            state.cta_group = cta_group if isinstance(cta_group, int) else None
        elif isinstance(cta_group, int) and state.cta_group is not None and cta_group != state.cta_group:
            raise ValueError(f"{op_name}: cta_group {cta_group} != {state.cta_group}")

    resolved_bufs: Dict[str, str] = {}
    for key in list(c.buffer_pre.keys()) + list(c.buffer_post.keys()):
        buf_name = _resolve_buffer_arg(op, key, g)
        if buf_name not in g.buffers:
            raise ValueError(f"{op_name}: unknown buffer '{buf_name}'")
        resolved_bufs[key] = buf_name

    # barrier presence checks (annotation-driven)
    if "bar" in op_args:
        bar = op_args["bar"]
        if bar not in g.barriers:
            raise ValueError(f"{op_name}: unknown barrier '{bar}'")
        if canonical != "mbarrier_init" and state.bar_state.get(bar) == BarrierState.UNINIT:
            raise ValueError(f"{op_name}: barrier '{bar}' used before init")

    for key, required in c.pre.items():
        bar = op_args.get(key)
        if bar is None:
            raise ValueError(f"{op_name}: missing barrier arg '{key}'")
        if bar not in g.barriers:
            raise ValueError(f"{op_name}: unknown barrier '{bar}'")
        if state.bar_state[bar] is not None and state.bar_state[bar] != required:
            raise ValueError(f"{op_name}: barrier {bar} state {state.bar_state[bar]} != {required}")

    for key, required in c.buffer_pre.items():
        buf = resolved_bufs[key]
        if state.buf_state[buf] is not None and state.buf_state[buf] != required:
            raise ValueError(f"{op_name}: buffer {buf} state {state.buf_state[buf]} != {required}")

    # Extra semantic checks derived from PTX ISA (best-effort for constants)
    if canonical == "tcgen05_alloc":
        cols = op_args.get("cols")
        if isinstance(cols, int):
            if cols < 32 or cols > GRAPH_TMEM_MAX_COLS:
                raise ValueError(f"{op_name}: cols {cols} out of range [32, {GRAPH_TMEM_MAX_COLS}]")
            if cols & (cols - 1) != 0:
                raise ValueError(f"{op_name}: cols {cols} must be power of 2")
            if state.last_alloc_cols is not None and cols > state.last_alloc_cols:
                raise ValueError(f"{op_name}: cols increased from {state.last_alloc_cols} to {cols}")
            state.last_alloc_cols = cols
    if canonical == "tcgen05_dealloc":
        cols = op_args.get("cols")
        if isinstance(cols, int) and state.last_alloc_cols is not None and cols != state.last_alloc_cols:
            raise ValueError(f"{op_name}: cols {cols} != last alloc {state.last_alloc_cols}")

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

    def _validate_desc_ref(desc_name: str, buf_hint: Optional[str] = None) -> None:
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
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                out[int(suffix)] = value
        return out

    def _validate_tmap_meta(tmap_meta: Dict[str, Any]) -> None:
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

    if canonical == "tcgen05_cp":
        shape = op_args.get("shape")
        tile = op_args.get("tile")
        if isinstance(shape, str):
            if (shape, tile) not in TCGEN05_CP_SHAPE_TILE:
                raise ValueError(f"{op_name}: shape/tile {(shape, tile)} not in {sorted(TCGEN05_CP_SHAPE_TILE)}")
        state.pending_tcgen_commit = True
        desc_name = op_args.get("desc")
        if desc_name is not None:
            desc = g.descriptors.get(str(desc_name))
            swizzle = _norm_swizzle(desc.meta.get("swizzle")) if desc else None
            major = str(desc.meta.get("major")).upper() if desc and desc.meta.get("major") else None
            allowed = _lookup_desc_lut("tcgen05_cp", op_args.get("shape"), op_args.get("tile"), swizzle, major)
            _validate_desc_ref(str(desc_name), buf_hint=op_args.get("smem_buf"))
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
        state.pending_tcgen_commit = True
        desc_a = op_args.get("desc_a")
        desc_b = op_args.get("desc_b")
        if desc_a is not None:
            desc = g.descriptors.get(str(desc_a))
            swizzle = _norm_swizzle(desc.meta.get("swizzle")) if desc else None
            major = str(desc.meta.get("major")).upper() if desc and desc.meta.get("major") else None
            allowed = _lookup_desc_lut("tcgen05_mma", op_args.get("shape"), op_args.get("tile"), swizzle, major)
            _validate_desc_ref(str(desc_a))
            if allowed is not None:
                if not isinstance(desc.meta.get("sbo"), int) or not isinstance(desc.meta.get("lbo"), int):
                    raise ValueError(f"{op_name}: descriptor '{desc_a}' requires numeric sbo/lbo for LUT check")
                pair = (int(desc.meta["sbo"]), int(desc.meta["lbo"]))
                if pair not in allowed:
                    raise ValueError(f"{op_name}: descriptor '{desc_a}' sbo/lbo {pair} not in LUT {sorted(allowed)}")
        if desc_b is not None:
            desc = g.descriptors.get(str(desc_b))
            swizzle = _norm_swizzle(desc.meta.get("swizzle")) if desc else None
            major = str(desc.meta.get("major")).upper() if desc and desc.meta.get("major") else None
            allowed = _lookup_desc_lut("tcgen05_mma", op_args.get("shape"), op_args.get("tile"), swizzle, major)
            _validate_desc_ref(str(desc_b))
            if allowed is not None:
                if not isinstance(desc.meta.get("sbo"), int) or not isinstance(desc.meta.get("lbo"), int):
                    raise ValueError(f"{op_name}: descriptor '{desc_b}' requires numeric sbo/lbo for LUT check")
                pair = (int(desc.meta["sbo"]), int(desc.meta["lbo"]))
                if pair not in allowed:
                    raise ValueError(f"{op_name}: descriptor '{desc_b}' sbo/lbo {pair} not in LUT {sorted(allowed)}")

    if canonical == "tma_gmem2smem":
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")
        for key in ("dst_align", "src_align"):
            align = op_args.get(key)
            if isinstance(align, int) and align < 16:
                raise ValueError(f"{op_name}: {key} {align} must be >= 16")
    if canonical.startswith("tma_"):
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")

    if canonical in {
        "tma_1d_gmem2smem",
        "tma_2d_gmem2smem",
        "tma_3d_gmem2smem",
        "tma_1d_gmem2smem_mcast",
        "tma_2d_gmem2smem_mcast",
        "tma_3d_gmem2smem_mcast",
    }:
        tmap = op_args.get("tmap")
        if tmap not in g.tmaps:
            raise ValueError(f"{op_name}: unknown tmap '{tmap}'")
        _validate_tmap_meta(g.tmaps[tmap])
        rank_required = {
            "tma_1d_gmem2smem": 1,
            "tma_2d_gmem2smem": 2,
            "tma_3d_gmem2smem": 3,
            "tma_1d_gmem2smem_mcast": 1,
            "tma_2d_gmem2smem_mcast": 2,
            "tma_3d_gmem2smem_mcast": 3,
        }[canonical]
        rank = g.tmaps[tmap].get("rank")
        if isinstance(rank, int) and rank != rank_required:
            raise ValueError(f"{op_name}: tmap '{tmap}' rank {rank} != {rank_required}")

        if canonical.endswith("_mcast"):
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

    if canonical in {"tma_1d_smem2gmem", "tma_2d_smem2gmem", "tma_3d_smem2gmem", "tma_store_out"}:
        tmap = op_args.get("tmap")
        if tmap not in g.tmaps:
            raise ValueError(f"{op_name}: unknown tmap '{tmap}'")
        _validate_tmap_meta(g.tmaps[tmap])
        rank = g.tmaps[tmap].get("rank")
        rank_required = {
            "tma_1d_smem2gmem": 1,
            "tma_2d_smem2gmem": 2,
            "tma_3d_smem2gmem": 3,
        }.get(canonical)
        if rank_required is not None and isinstance(rank, int) and rank != rank_required:
            raise ValueError(f"{op_name}: tmap '{tmap}' rank {rank} != {rank_required}")
        if canonical == "tma_store_out":
            req = op_args.get("rank")
            if isinstance(req, int) and isinstance(rank, int) and req != rank:
                raise ValueError(f"{op_name}: rank {req} != tmap rank {rank}")

    if canonical == "mbarrier_init":
        count = op_args.get("count")
        if isinstance(count, int) and count <= 0:
            raise ValueError(f"{op_name}: count must be > 0")
    if canonical in ("mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"):
        phase = op_args.get("phase")
        if isinstance(phase, int) and phase not in (0, 1):
            raise ValueError(f"{op_name}: phase {phase} must be 0 or 1")
    if canonical in ("mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta"):
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")

    if canonical == "tcgen05_ld":
        shape = op_args.get("shape")
        num = op_args.get("num")
        if isinstance(shape, str) and shape not in TCGEN05_LD_SHAPES:
            raise ValueError(f"{op_name}: shape {shape} not in {sorted(TCGEN05_LD_SHAPES)}")
        if isinstance(num, int) and num not in TCGEN05_NUM_VALUES:
            raise ValueError(f"{op_name}: num {num} not in {sorted(TCGEN05_NUM_VALUES)}")
        if "warp_id" not in op_args or "lane_id" not in op_args:
            raise ValueError(f"{op_name}: requires warp_id and lane_id metadata for tcgen05.ld")
        lane_val = op_args.get("lane_id")
        if isinstance(lane_val, str) and lane_val.lower() in {"elect", "leader"}:
            raise ValueError(f"{op_name}: lane_id=elect/leader invalid for tcgen05.ld")
        state.pending_ld = True
    if canonical == "tcgen05_st":
        shape = op_args.get("shape")
        num = op_args.get("num")
        if isinstance(shape, str) and shape not in TCGEN05_LD_SHAPES:
            raise ValueError(f"{op_name}: shape {shape} not in {sorted(TCGEN05_LD_SHAPES)}")
        if isinstance(num, int) and num not in TCGEN05_NUM_VALUES:
            raise ValueError(f"{op_name}: num {num} not in {sorted(TCGEN05_NUM_VALUES)}")
        if "warp_id" not in op_args or "lane_id" not in op_args:
            raise ValueError(f"{op_name}: requires warp_id and lane_id metadata for tcgen05.st")
        lane_val = op_args.get("lane_id")
        if isinstance(lane_val, str) and lane_val.lower() in {"elect", "leader"}:
            raise ValueError(f"{op_name}: lane_id=elect/leader invalid for tcgen05.st")
        state.pending_st = True
    if canonical == "tcgen05_wait_ld":
        if state.pending_ld is False:
            raise ValueError(f"{op_name}: wait_ld without prior ld")
        if state.pending_ld is True:
            state.pending_ld = False
        else:
            state.pending_ld = None
    if canonical == "tcgen05_wait_st":
        if state.pending_st is False:
            raise ValueError(f"{op_name}: wait_st without prior st")
        if state.pending_st is True:
            state.pending_st = False
        else:
            state.pending_st = None
    if canonical == "tcgen05_wait":
        if state.pending_ld is False and state.pending_st is False:
            raise ValueError(f"{op_name}: wait without prior ld/st")
        state.pending_ld = False if state.pending_ld is True else state.pending_ld
        state.pending_st = False if state.pending_st is True else state.pending_st
    if canonical in ("tcgen05_commit", "tcgen05_commit_mcast"):
        if state.pending_tcgen_commit is False:
            raise ValueError(f"{op_name}: commit without prior tcgen05 cp/mma")

    if canonical == "mbarrier_fence_init_release":
        state.cluster_init_fenced = True
    if canonical == "barrier_cluster_wait":
        state.cluster_sync_done = True

    if "bar" in op_args:
        bar = op_args["bar"]
        if bar in g.barriers and g.barriers[bar].scope == "cluster":
            if state.cluster_init_fenced is False:
                raise ValueError(f"{op_name}: cluster barrier '{bar}' used before fence.mbarrier_init.release.cluster")
            if state.cluster_sync_done is False:
                raise ValueError(f"{op_name}: cluster barrier '{bar}' used before barrier.cluster.wait")

    def _add_optional(cur: Optional[int], delta: int) -> Optional[int]:
        if cur is None:
            return None
        return cur + delta

    # minimal barrier state transitions
    if canonical == "mbarrier_init":
        bar = op_args.get("bar")
        if bar in state.bar_state:
            state.bar_state[bar] = BarrierState.INIT
            count = op_args.get("count")
            if isinstance(count, int):
                prev = state.bar_init_count.get(bar)
                if prev is not None and prev != count:
                    raise ValueError(f"{op_name}: barrier '{bar}' count {count} != {prev}")
                state.bar_init_count[bar] = count
            else:
                state.bar_init_count[bar] = None
            state.bar_phase[bar] = 0
            state.bar_arrivals[bar] = 0
            state.bar_expected_bytes[bar] = 0
            state.bar_completed_bytes[bar] = 0

    if canonical in ("mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta", "tcgen05_commit", "tcgen05_commit_mcast"):
        bar = op_args.get("bar")
        if bar in state.bar_arrivals:
            state.bar_arrivals[bar] = _add_optional(state.bar_arrivals.get(bar), 1)
            count = state.bar_init_count.get(bar)
            arrivals = state.bar_arrivals.get(bar)
            if isinstance(count, int) and isinstance(arrivals, int) and arrivals > count:
                raise ValueError(f"{op_name}: barrier '{bar}' arrivals {arrivals} > count {count}")
        if canonical in ("mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta"):
            size = op_args.get("size")
            if bar in state.bar_expected_bytes:
                if isinstance(size, int):
                    state.bar_expected_bytes[bar] = _add_optional(state.bar_expected_bytes.get(bar), size)
                else:
                    state.bar_expected_bytes[bar] = None
            completed = state.bar_completed_bytes.get(bar)
            if isinstance(size, int) and isinstance(completed, int) and completed > size:
                raise ValueError(f"{op_name}: barrier '{bar}' completed {completed} > expected {size}")

    if canonical in (
        "tma_gmem2smem",
        "tma_1d_gmem2smem",
        "tma_2d_gmem2smem",
        "tma_3d_gmem2smem",
        "tma_1d_gmem2smem_mcast",
        "tma_2d_gmem2smem_mcast",
        "tma_3d_gmem2smem_mcast",
    ):
        bar = op_args.get("bar")
        if bar in state.bar_completed_bytes:
            size = op_args.get("size")
            if isinstance(size, int):
                state.bar_completed_bytes[bar] = _add_optional(state.bar_completed_bytes.get(bar), size)
            else:
                state.bar_completed_bytes[bar] = None

    if canonical in ("mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"):
        bar = op_args.get("bar")
        phase = op_args.get("phase")
        if bar in state.bar_phase:
            bar_phase = state.bar_phase.get(bar)
            if isinstance(phase, int) and bar_phase is not None and phase != bar_phase:
                raise ValueError(f"{op_name}: barrier '{bar}' phase {phase} != {bar_phase}")
            if bar_phase is None and isinstance(phase, int):
                state.bar_phase[bar] = phase
        count = state.bar_init_count.get(bar)
        arrivals = state.bar_arrivals.get(bar)
        if isinstance(count, int) and isinstance(arrivals, int) and arrivals > count:
            raise ValueError(f"{op_name}: barrier '{bar}' arrivals {arrivals} > count {count}")
        expected = state.bar_expected_bytes.get(bar)
        completed = state.bar_completed_bytes.get(bar)
        if isinstance(expected, int) and isinstance(completed, int) and completed > expected:
            raise ValueError(f"{op_name}: barrier '{bar}' completed {completed} > expected {expected}")
        if bar in state.bar_arrivals:
            state.bar_arrivals[bar] = 0
            state.bar_expected_bytes[bar] = 0
            state.bar_completed_bytes[bar] = 0
        if bar in state.bar_phase and isinstance(state.bar_phase[bar], int):
            state.bar_phase[bar] = 1 - int(state.bar_phase[bar])

    for key, new_state in c.post.items():
        bar = op_args[key]
        state.bar_state[bar] = new_state

    for key, new_state in c.buffer_post.items():
        buf = resolved_bufs[key]
        state.buf_state[buf] = new_state


def _clone_state(state: ValidationState) -> ValidationState:
    return ValidationState(
        bar_state=dict(state.bar_state),
        buf_state=dict(state.buf_state),
        bar_init_count=dict(state.bar_init_count),
        bar_arrivals=dict(state.bar_arrivals),
        bar_phase=dict(state.bar_phase),
        bar_expected_bytes=dict(state.bar_expected_bytes),
        bar_completed_bytes=dict(state.bar_completed_bytes),
        cluster_init_fenced=state.cluster_init_fenced,
        cluster_sync_done=state.cluster_sync_done,
        cluster_ctas=state.cluster_ctas,
        pending_ld=state.pending_ld,
        pending_st=state.pending_st,
        pending_tcgen_commit=state.pending_tcgen_commit,
        cta_group=state.cta_group,
        last_alloc_cols=state.last_alloc_cols,
    )


def _merge_optional(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a == b else None


def _merge_dict(a: Dict[str, Optional[Any]], b: Dict[str, Optional[Any]]) -> Dict[str, Optional[Any]]:
    merged: Dict[str, Optional[Any]] = {}
    for key in set(a.keys()) | set(b.keys()):
        merged[key] = _merge_optional(a.get(key), b.get(key))
    return merged


def _merge_states(a: ValidationState, b: ValidationState) -> ValidationState:
    return ValidationState(
        bar_state=_merge_dict(a.bar_state, b.bar_state),
        buf_state=_merge_dict(a.buf_state, b.buf_state),
        bar_init_count=_merge_dict(a.bar_init_count, b.bar_init_count),
        bar_arrivals=_merge_dict(a.bar_arrivals, b.bar_arrivals),
        bar_phase=_merge_dict(a.bar_phase, b.bar_phase),
        bar_expected_bytes=_merge_dict(a.bar_expected_bytes, b.bar_expected_bytes),
        bar_completed_bytes=_merge_dict(a.bar_completed_bytes, b.bar_completed_bytes),
        cluster_init_fenced=_merge_optional(a.cluster_init_fenced, b.cluster_init_fenced),
        cluster_sync_done=_merge_optional(a.cluster_sync_done, b.cluster_sync_done),
        cluster_ctas=_merge_optional(a.cluster_ctas, b.cluster_ctas),
        pending_ld=_merge_optional(a.pending_ld, b.pending_ld),
        pending_st=_merge_optional(a.pending_st, b.pending_st),
        pending_tcgen_commit=_merge_optional(a.pending_tcgen_commit, b.pending_tcgen_commit),
        cta_group=_merge_optional(a.cta_group, b.cta_group),
        last_alloc_cols=_merge_optional(a.last_alloc_cols, b.last_alloc_cols),
    )


def _validate_nodes(nodes: List[Node], g: Graph, state: ValidationState) -> None:
    for node in nodes:
        if node.kind == "KernelStart":
            state.bar_state = {name: BarrierState.UNINIT for name in g.barriers}
            state.buf_state = {name: BufferState.EMPTY for name in g.buffers}
            state.bar_init_count = {name: None for name in g.barriers}
            state.bar_arrivals = {name: None for name in g.barriers}
            state.bar_phase = {name: None for name in g.barriers}
            state.bar_expected_bytes = {name: None for name in g.barriers}
            state.bar_completed_bytes = {name: None for name in g.barriers}
            state.cluster_init_fenced = False
            state.cluster_sync_done = False
            state.cluster_ctas = None
            state.pending_ld = False
            state.pending_st = False
            state.pending_tcgen_commit = False
            state.cta_group = None
            state.last_alloc_cols = None
            smem_static = node.args.get("smem_bytes") or node.args.get("smem_static")
            smem_dynamic = node.args.get("smem_dynamic")
            cluster_ctas = node.args.get("cluster_ctas")
            if isinstance(cluster_ctas, int):
                state.cluster_ctas = cluster_ctas
            else:
                dim_x = node.args.get("cluster_dim_x")
                dim_y = node.args.get("cluster_dim_y")
                dim_z = node.args.get("cluster_dim_z")
                if all(isinstance(v, int) for v in (dim_x, dim_y, dim_z)):
                    state.cluster_ctas = int(dim_x) * int(dim_y) * int(dim_z)
            total_smem = None
            if isinstance(smem_static, int) and isinstance(smem_dynamic, int):
                total_smem = smem_static + smem_dynamic
            elif isinstance(smem_static, int):
                total_smem = smem_static
            if total_smem is not None and total_smem > GRAPH_SMEM_LIMIT_BYTES:
                raise ValueError(
                    f"KernelStart: smem {total_smem} > assumed limit {GRAPH_SMEM_LIMIT_BYTES} bytes"
                )
            continue
        if node.kind == "KernelEnd":
            # ensure tmem is deallocated before leaving kernel
            for buf, st in state.buf_state.items():
                if st is not None and st != BufferState.EMPTY:
                    raise ValueError(f"Kernel end: buffer {buf} not deallocated ({st})")
            if state.pending_ld is True:
                raise ValueError("Kernel end: pending tcgen05.ld without wait_ld")
            if state.pending_st is True:
                raise ValueError("Kernel end: pending tcgen05.st without wait_st")
            continue

        if node.kind == "Block":
            _validate_nodes(node.children, g, state)
            continue

        if node.kind == "If":
            if "cond" not in node.args:
                raise ValueError("If node missing 'cond'")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            else_node = next((c for c in node.children if c.kind == "Else"), None)

            s1 = _clone_state(state)
            if then_node:
                _validate_nodes(then_node.children, g, s1)

            s2 = _clone_state(state)
            if else_node:
                _validate_nodes(else_node.children, g, s2)

            merged = _merge_states(s1, s2)
            state.bar_state = merged.bar_state
            state.buf_state = merged.buf_state
            state.bar_init_count = merged.bar_init_count
            state.bar_arrivals = merged.bar_arrivals
            state.bar_phase = merged.bar_phase
            state.bar_expected_bytes = merged.bar_expected_bytes
            state.bar_completed_bytes = merged.bar_completed_bytes
            state.cluster_init_fenced = merged.cluster_init_fenced
            state.cluster_sync_done = merged.cluster_sync_done
            state.pending_ld = merged.pending_ld
            state.pending_st = merged.pending_st
            state.pending_tcgen_commit = merged.pending_tcgen_commit
            state.cta_group = merged.cta_group
            state.last_alloc_cols = merged.last_alloc_cols
            continue

        if node.kind in ("Raw", "LoadInline", "Launch"):
            # Raw nodes are opaque; validation happens only on annotated ops.
            continue
        if node.kind == "Op":
            _validate_op(node, g, state)
            continue

        if node.kind == "For":
            if "iters" not in node.args or "var" not in node.args:
                raise ValueError("For node requires 'var' and 'iters'")
            iters = int(node.args["iters"])
            for _ in range(iters):
                _validate_nodes(node.children, g, state)
            continue

        if node.kind in ("Then", "Else"):
            _validate_nodes(node.children, g, state)
            continue

        _validate_op(node, g, state)


def _collect_tmaps(nodes: List[Node], g: Graph) -> None:
    for node in nodes:
        if node.kind == "Op":
            op_name = str(node.args.get("op", ""))
            op_args = dict(node.args.get("op_args", {}))
            canonical = _canonical_op_name(op_name)
            if canonical in {"cute_tmap", "tmap_create"}:
                g.add_tmap(str(op_args["name"]), op_args)
        if node.children:
            _collect_tmaps(node.children, g)


def _dtype_size_bytes(dtype: str) -> Optional[int]:
    key = dtype.lower()
    if key in {"f16", "half", "fp16"}:
        return 2
    if key in {"bf16", "bfloat16"}:
        return 2
    if key in {"f32", "float", "fp32"}:
        return 4
    if key in {"f64", "double", "fp64"}:
        return 8
    if key in {"i8", "int8", "u8", "uint8"}:
        return 1
    if key in {"i16", "int16", "u16", "uint16"}:
        return 2
    if key in {"i32", "int32", "u32", "uint32", "int"}:
        return 4
    if key in {"i64", "int64", "u64", "uint64"}:
        return 8
    return None


def _estimate_smem_bytes(g: Graph) -> Optional[int]:
    total = 0
    for buf in g.buffers.values():
        if buf.space != MemSpace.SMEM:
            continue
        meta = buf.meta
        explicit = meta.get("bytes") or meta.get("size") or meta.get("smem_bytes")
        if isinstance(explicit, int):
            total += explicit
            continue
        size = _dtype_size_bytes(buf.dtype)
        if size is None:
            return None
        if not buf.shape or any(not isinstance(dim, int) for dim in buf.shape):
            return None
        elems = 1
        for dim in buf.shape:
            elems *= dim
        total += elems * size
    return total


def validate_graph(g: Graph) -> None:
    _collect_tmaps(g.sections.get("host", []), g)
    validate_graph_ptx_spec(g)
    smem_bytes = _estimate_smem_bytes(g)
    if smem_bytes is not None and smem_bytes > GRAPH_SMEM_LIMIT_BYTES:
        raise ValueError(f"SMEM usage {smem_bytes} exceeds assumed limit {GRAPH_SMEM_LIMIT_BYTES} bytes")
    state = ValidationState(
        bar_state={name: BarrierState.UNINIT for name in g.barriers},
        buf_state={name: BufferState.EMPTY for name in g.buffers},
        bar_init_count={name: None for name in g.barriers},
        bar_arrivals={name: None for name in g.barriers},
        bar_phase={name: None for name in g.barriers},
        bar_expected_bytes={name: None for name in g.barriers},
        bar_completed_bytes={name: None for name in g.barriers},
        cluster_init_fenced=False,
        cluster_sync_done=False,
        cluster_ctas=None,
    )
    _validate_nodes(g.sections.get("device", []), g, state)
    validate_graph_protocol(g, strict=bool(g.meta.get("strict_protocol", False)))
