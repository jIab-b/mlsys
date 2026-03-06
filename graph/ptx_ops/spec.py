"""PTX ISA ground truth: known ops, arg specs, hardware constants."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Canonical op names: real PTX/hardware ops only.
KNOWN_OPS = {
    # tcgen05 core
    "tcgen05_alloc", "tcgen05_dealloc", "tcgen05_cp", "tcgen05_mma",
    "tcgen05_ld", "tcgen05_st", "tcgen05_commit", "tcgen05_commit_mcast",
    "tcgen05_wait", "tcgen05_wait_ld", "tcgen05_wait_st",
    "tcgen05_fence", "tcgen05_fence_before_thread_sync", "tcgen05_fence_after_thread_sync",
    # mbarrier
    "mbarrier_init", "mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta",
    "mbarrier_wait", "mbarrier_wait_ticks", "mbarrier_wait_relaxed",
    "mbarrier_fence_init_release",
    # cluster sync
    "barrier_cluster_arrive", "barrier_cluster_wait",
    # tma (cp.async.bulk.tensor) — collapsed, rank is an arg
    "tma_load", "tma_load_mcast", "tma_store",
    # host-side metadata ops
    "tmap_create", "cute_tmap", "cta_group_set",
}

OP_ARG_SPECS: Dict[str, Dict[str, Any]] = {
    "tcgen05_alloc": {
        "required": {"tmem", "cols"},
        "ints": {"cols"},
        "optional": {"cta_group", "scope"},
    },
    "tcgen05_dealloc": {
        "required": {"tmem", "cols"},
        "ints": {"cols"},
        "optional": {"cta_group", "scope"},
    },
    "tcgen05_cp": {
        "required": {"tmem"},
        "optional": {"cta_group", "tmem_offset", "cols", "smem_desc"},
        "ints": {"tmem_offset", "cols"},
    },
    "tcgen05_mma": {
        "required": {"tmem"},
        "optional": {"cta_group", "idesc", "smem_desc_a", "smem_desc_b"},
    },
    "tcgen05_ld": {
        "required": {"tmem"},
        "optional": {"cta_group"},
    },
    "tcgen05_st": {
        "required": {"tmem"},
        "optional": {"cta_group"},
    },
    "tcgen05_commit": {
        "required": {"bar"},
        "optional": {"cta_group"},
    },
    "tcgen05_commit_mcast": {
        "required": {"bar", "cta_mask"},
        "ints": {"cta_mask"},
        "optional": {"cta_group"},
    },
    "tcgen05_wait_ld": {
        "required": set(),
        "optional": set(),
    },
    "tcgen05_wait_st": {
        "required": set(),
        "optional": set(),
    },
    "tcgen05_fence_after_thread_sync": {
        "required": set(),
        "optional": set(),
    },
    "mbarrier_init": {
        "required": {"bar", "count"},
        "ints": {"count"},
        "optional": {"scope"},
    },
    "mbarrier_arrive_expect_tx": {
        "required": {"bar", "size"},
        "ints": {"size"},
        "optional": {"scope"},
    },
    "mbarrier_arrive_expect_tx_cta": {
        "required": {"bar", "size"},
        "ints": {"size"},
        "optional": {"scope"},
    },
    "mbarrier_wait": {
        "required": {"bar", "phase"},
        "ints": {"phase"},
        "optional": {"scope"},
    },
    "mbarrier_wait_relaxed": {
        "required": {"bar", "phase"},
        "ints": {"phase"},
        "optional": {"scope"},
    },
    "mbarrier_wait_ticks": {
        "required": {"bar", "phase"},
        "ints": {"phase", "ticks"},
        "optional": {"scope", "ticks"},
    },
    "tma_load": {
        "required": {"bar", "tmap"},
        "optional": {"rank", "size"},
        "ints": {"rank", "size"},
    },
    "tma_load_mcast": {
        "required": {"bar", "tmap", "cta_mask"},
        "optional": {"rank"},
        "ints": {"rank", "cta_mask"},
    },
    "tma_store": {
        "required": {"tmap"},
        "optional": {"rank"},
        "ints": {"rank"},
    },
    "barrier_cluster_arrive": {
        "required": set(),
        "optional": set(),
    },
    "barrier_cluster_wait": {
        "required": set(),
        "optional": set(),
    },
    "tmap_create": {
        "required": {"name", "rank", "dtype", "swizzle", "interleave"},
        "optional": {"global_dim0", "global_dim1", "global_dim2", "global_stride0", "global_stride1", "global_stride2"},
        "ints": {"rank", "global_dim0", "global_dim1", "global_dim2", "global_stride0", "global_stride1", "global_stride2"},
    },
    "cute_tmap": {
        "required": {"name"},
        "optional": {"rank", "global_height", "global_width", "shared_height", "shared_width"},
        "ints": {"rank", "global_height", "global_width", "shared_height", "shared_width"},
    },
    "cta_group_set": {
        "required": {"value"},
        "optional": set(),
        "ints": {"value"},
    },
}

ISSUE_SCOPES = {"one_thread", "one_warp", "all_warps", "host"}
BARRIER_SCOPES = {"cta", "cluster"}

GRAPH_SMEM_LIMIT_BYTES = 227 * 1024 - 1024
GRAPH_TMEM_MAX_COLS = 512
CTA_MASK_BITS = 16

# tcgen05 descriptor / shape constraints (from PTX ISA docs).
TCGEN05_SWIZZLE_VALID = {"none", "32b", "64b", "128b", "128b32a"}
TCGEN05_SWIZZLE_ALIGN_BYTES = {"32b": 256, "64b": 512, "128b": 1024, "128b32a": 1024}
TCGEN05_NUM_VALUES = {1, 2, 4, 8, 16, 32, 64, 128}
TCGEN05_LD_SHAPES = {"16x64b", "16x128b", "16x256b", "32x32b", "16x32bx2"}
TCGEN05_CP_SHAPE_TILE = {
    ("32x128b", "warpx4"),
    ("128x128b", None),
    ("128x256b", None),
    ("4x256b", None),
    ("64x128b", None),
}
TCGEN05_MMA_SHAPES = {
    "mxf4nvf4.block16",
    "mxf4.block16",
    "f16.ss",
    "f16.ts",
    "bf16.ss",
    "bf16.ts",
    "ws.f16.ts",
}

PTX_TCGEN05_CP_SHAPE_TILE = TCGEN05_CP_SHAPE_TILE
PTX_TCGEN05_MMA_SHAPES = TCGEN05_MMA_SHAPES
PTX_TCGEN05_NO_TRANSPOSE_KINDS = {"mxf4", "mxf4nvf4"}

TMA_INTERLEAVE_SET = {"none", "16b", "32b"}
TMA_SWIZZLE_SET = {
    "none", "32b", "64b", "128b",
    "128b_atom_32b", "128b_atom_32b_flip_8b", "128b_atom_64b",
}
TMA_DTYPE_ELEMENT_SIZE_BYTES = {
    "16u4_align8b": 0.5, "16u4_align16b": 0.5, "16u6_align16b": 0.75,
    "bf16": 2.0, "f16": 2.0, "f32": 4.0,
}
TMA_DTYPE_STRIDE32 = {"16u4_align16b", "16u6_align16b"}
TMA_DTYPE_SWIZZLE_ALLOWED = {
    "16u6_align16b": {"none", "128b", "128b_atom_32b", "128b_atom_64b"},
    "16u4_align16b": {"none", "128b", "128b_atom_32b"},
}

TCGEN_DESC_SBO_LBO_LUT: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], set[tuple[int, int]]] = {
    ("tcgen05_cp", "32x128b", "warpx4", "none", "K"): {(128, 1)},
    ("tcgen05_mma", "mxf4nvf4.block16", None, "128b", "K"): {(1024, 1)},
}

# Simple descriptor objects for static legality checks.
TCGEN05_SMEM_DESC_LEGAL = {
    "keys": {"raw", "start_enc", "ld_enc", "sd_enc", "base_offset", "ld_mode", "swizzle_code", "fixed_46_48"},
    "ranges": {
        "raw": (0, 0xFFFFFFFFFFFFFFFF),
        "start_enc": (0, 0x3FFF),
        "ld_enc": (0, 0x3FFF),
        "sd_enc": (0, 0x3FFF),
        "base_offset": (0, 0x7),
        "ld_mode": (0, 1),
        "swizzle_code": (0, 7),
        "fixed_46_48": (0b001, 0b001),
    },
    "invalid_swizzle_code": {3, 5, 7},
    "valid_swizzle_code": {0, 1, 2, 4, 6},
}

TCGEN05_IDESC_LEGAL = {
    "keys": {
        "raw",
        # table-42 style fields
        "sparsity_selector", "sparsity", "saturate", "dtype_d", "reserved_6",
        "atype", "btype", "negate_a", "negate_b", "transpose_a", "transpose_b",
        "n_over_8", "reserved_23", "m_over_16", "reserved_29", "ws_b_reuse_shift",
        # table-43/44 extra fields
        "reserved_0_1", "reserved_3", "b_scale_data_id", "scale_matrix_type",
        "reserved_24_26", "m_over_128", "a_scale_data_id", "reserved_31",
        "reserved_12", "k_dim_selector",
    },
    "ranges": {
        "raw": (0, 0xFFFFFFFF),
        "sparsity_selector": (0, 3),
        "sparsity": (0, 1),
        "saturate": (0, 1),
        "dtype_d": (0, 3),
        "reserved_6": (0, 0),
        "atype": (0, 7),
        "btype": (0, 7),
        "negate_a": (0, 1),
        "negate_b": (0, 1),
        "transpose_a": (0, 1),
        "transpose_b": (0, 1),
        "n_over_8": (0, 63),
        "reserved_23": (0, 0),
        "m_over_16": (0, 31),
        "reserved_29": (0, 0),
        "ws_b_reuse_shift": (0, 3),
        "reserved_0_1": (0, 0),
        "reserved_3": (0, 0),
        "b_scale_data_id": (0, 3),
        "scale_matrix_type": (0, 1),
        "reserved_24_26": (0, 0),
        "m_over_128": (0, 3),
        "a_scale_data_id": (0, 3),
        "reserved_31": (0, 0),
        "reserved_12": (0, 0),
        "k_dim_selector": (0, 1),
    },
}


def _canonical_op_name(kind: str) -> str:
    """Resolve op name to canonical form. Simple lookup, no fuzzy matching."""
    if kind in KNOWN_OPS:
        return kind
    return kind
