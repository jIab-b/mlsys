from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
PTX_LIB = REPO_ROOT / "ptx_lib"


# Canonical op names: all known base ops.
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
    "barrier_cluster_arrive", "barrier_cluster_wait",
    # tma / cp.async bulk
    "tma_gmem2smem", "tma_1d_gmem2smem", "tma_2d_gmem2smem", "tma_3d_gmem2smem",
    "tma_1d_gmem2smem_mcast", "tma_2d_gmem2smem_mcast", "tma_3d_gmem2smem_mcast",
    "tma_1d_smem2gmem", "tma_2d_smem2gmem", "tma_3d_smem2gmem",
    "tma_store_out", "tmap_create",
    "cp_async_bulk_prefetch", "cp_async_bulk_prefetch_1d",
    "cp_async_bulk_prefetch_2d", "cp_async_bulk_prefetch_3d",
    # ptx common helpers
    "ptx_laneid", "ptx_activemask", "ptx_elect_one_sync", "ptx_elect_sync",
    "ptx_bar_sync",
    # host-side metadata ops
    "cute_tmap", "cta_group_set",
    "persistent_loop_begin", "persistent_loop_end",
}

TCGEN_PREFIX_CONTRACTS: Tuple[Tuple[str, str], ...] = (
    ("tcgen05_alloc", "tcgen05_alloc"),
    ("tcgen05_dealloc", "tcgen05_dealloc"),
    ("tcgen05_cp", "tcgen05_cp"),
    ("tcgen05_mma", "tcgen05_mma"),
    ("tcgen05_ld", "tcgen05_ld"),
    ("tcgen05_st", "tcgen05_st"),
    ("tcgen05_commit", "tcgen05_commit"),
    ("tcgen05_wait", "tcgen05_wait"),
    ("tcgen05_fence", "tcgen05_fence"),
    ("mbarrier_", "mbarrier_init"),
    ("tma_", "tma_1d_gmem2smem"),
    ("ptx_", "ptx_laneid"),
    ("barrier_cluster_arrive", "barrier_cluster_arrive"),
    ("barrier_cluster_wait", "barrier_cluster_wait"),
)

OP_ALIASES: Dict[str, str] = {
    "tcgen05_cp_nvfp4": "tcgen05_cp",
    "tcgen05_mma_nvfp4": "tcgen05_mma",
    "tcgen05_ld_32x32bx128": "tcgen05_ld",
    "tcgen05_ld_32x32bx64": "tcgen05_ld",
    "tcgen05_ld_32x32bx32": "tcgen05_ld",
    "tcgen05_ld_16x256bx16": "tcgen05_ld",
    "tcgen05_ld_16x256bx8": "tcgen05_ld",
    "tcgen05_ld_16x256bx4": "tcgen05_ld",
    "ptx_bar_sync": "ptx_bar_sync",
    "tma_1d_smem2gmem": "tma_1d_smem2gmem",
    "tma_2d_smem2gmem": "tma_2d_smem2gmem",
    "tma_3d_smem2gmem": "tma_3d_smem2gmem",
}


def _extract_ptx_functions(header_text: str) -> set[str]:
    names: set[str] = set()
    for line in header_text.splitlines():
        if "PTX_DEVICE" not in line and "__device__" not in line:
            continue
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            names.add(m.group(1))
    return names


def _infer_ptx_base(name: str) -> Optional[str]:
    if name in KNOWN_OPS:
        return name
    if name.startswith("tcgen05_commit_mcast"):
        return "tcgen05_commit_mcast"
    if name.startswith("tcgen05_commit"):
        return "tcgen05_commit"
    if name.startswith("tcgen05_wait_ld"):
        return "tcgen05_wait_ld"
    if name.startswith("tcgen05_wait_st"):
        return "tcgen05_wait_st"
    if name.startswith("tcgen05_wait"):
        return "tcgen05_wait"
    if name.startswith("tcgen05_fence_before_thread_sync"):
        return "tcgen05_fence_before_thread_sync"
    if name.startswith("tcgen05_fence_after_thread_sync"):
        return "tcgen05_fence_after_thread_sync"
    if name.startswith("tcgen05_fence"):
        return "tcgen05_fence"
    if name.startswith("tcgen05_alloc"):
        return "tcgen05_alloc"
    if name.startswith("tcgen05_dealloc"):
        return "tcgen05_dealloc"
    if name.startswith("tcgen05_cp"):
        return "tcgen05_cp"
    if name.startswith("tcgen05_mma"):
        return "tcgen05_mma"
    if name.startswith("tcgen05_ld"):
        return "tcgen05_ld"
    if name.startswith("tcgen05_st"):
        return "tcgen05_st"
    if name.startswith("mbarrier_arrive_expect_tx_cta"):
        return "mbarrier_arrive_expect_tx_cta"
    if name.startswith("mbarrier_arrive_expect_tx"):
        return "mbarrier_arrive_expect_tx"
    if name.startswith("mbarrier_wait_relaxed"):
        return "mbarrier_wait_relaxed"
    if name.startswith("mbarrier_wait_ticks"):
        return "mbarrier_wait_ticks"
    if name.startswith("mbarrier_wait"):
        return "mbarrier_wait"
    if name.startswith("mbarrier_fence_init_release"):
        return "mbarrier_fence_init_release"
    if name.startswith("mbarrier_init"):
        return "mbarrier_init"
    if name.startswith("barrier_cluster_arrive"):
        return "barrier_cluster_arrive"
    if name.startswith("barrier_cluster_wait"):
        return "barrier_cluster_wait"
    if name.startswith("tma_"):
        if "smem2gmem" in name:
            if name.startswith("tma_3d"):
                return "tma_3d_smem2gmem"
            if name.startswith("tma_2d"):
                return "tma_2d_smem2gmem"
            if name.startswith("tma_1d"):
                return "tma_1d_smem2gmem"
            return "tma_store_out"
        if name in KNOWN_OPS:
            return name
        if "3d" in name:
            return "tma_3d_gmem2smem"
        if "2d" in name:
            return "tma_2d_gmem2smem"
        if "1d" in name:
            return "tma_1d_gmem2smem"
        return "tma_gmem2smem"
    if name.startswith("ptx_"):
        if name in KNOWN_OPS:
            return name
        return "ptx_laneid"
    return None


def _load_ptx_aliases() -> Dict[str, str]:
    if not PTX_LIB.exists():
        return {}
    aliases: Dict[str, str] = {}
    for header in PTX_LIB.glob("*.cuh"):
        names = _extract_ptx_functions(header.read_text())
        for name in names:
            base = _infer_ptx_base(name)
            if base and base != name:
                aliases.setdefault(name, base)
    return aliases


OP_ALIASES.update(_load_ptx_aliases())

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
        "optional": {"cta_group", "tmem_offset", "cols"},
        "ints": {"tmem_offset", "cols"},
    },
    "tcgen05_mma": {
        "required": {"tmem"},
        "optional": {"cta_group"},
    },
    "tcgen05_ld": {
        "required": {"tmem"},
        "optional": {"cta_group"},
    },
    "tcgen05_wait_ld": {
        "required": set(),
        "optional": set(),
    },
    "tcgen05_commit": {
        "required": {"bar"},
        "optional": {"cta_group"},
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
    "tma_gmem2smem": {
        "required": {"bar", "size"},
        "ints": {"size"},
        "optional": {"dst_align", "src_align"},
    },
    "tma_3d_gmem2smem": {
        "required": {"bar", "tmap"},
        "optional": {"dim"},
    },
    "tma_1d_gmem2smem": {
        "required": {"bar", "tmap"},
        "optional": {"dim"},
    },
    "tma_2d_gmem2smem": {
        "required": {"bar", "tmap"},
        "optional": {"dim"},
    },
    "tma_1d_gmem2smem_mcast": {
        "required": {"bar", "tmap", "cta_mask"},
        "optional": {"dim"},
        "ints": {"cta_mask"},
    },
    "tma_2d_gmem2smem_mcast": {
        "required": {"bar", "tmap", "cta_mask"},
        "optional": {"dim"},
        "ints": {"cta_mask"},
    },
    "tma_3d_gmem2smem_mcast": {
        "required": {"bar", "tmap", "cta_mask"},
        "optional": {"dim"},
        "ints": {"cta_mask"},
    },
    "tma_1d_smem2gmem": {
        "required": {"tmap"},
        "optional": {"x"},
        "ints": {"x"},
    },
    "tma_2d_smem2gmem": {
        "required": {"tmap"},
        "optional": {"x", "y"},
        "ints": {"x", "y"},
    },
    "tma_3d_smem2gmem": {
        "required": {"tmap"},
        "optional": {"x", "y", "z"},
        "ints": {"x", "y", "z"},
    },
    "tma_store_out": {
        "required": {"tmap"},
        "optional": {"x", "y", "z", "rank"},
        "ints": {"x", "y", "z", "rank"},
    },
    "tmap_create": {
        "required": {"name", "rank", "dtype", "swizzle", "interleave"},
        "optional": {"global_dim0", "global_dim1", "global_dim2", "global_stride0", "global_stride1", "global_stride2"},
        "ints": {"rank", "global_dim0", "global_dim1", "global_dim2", "global_stride0", "global_stride1", "global_stride2"},
    },
    "tcgen05_commit_mcast": {
        "required": {"bar", "cta_mask"},
        "ints": {"cta_mask"},
        "optional": {"cta_group"},
    },
    "barrier_cluster_arrive": {
        "required": set(),
        "optional": set(),
    },
    "barrier_cluster_wait": {
        "required": set(),
        "optional": set(),
    },
    "cp_async_bulk_prefetch": {
        "required": {"addr", "size"},
        "ints": {"size"},
        "optional": set(),
    },
    "cp_async_bulk_prefetch_1d": {
        "required": {"tmap", "x"},
        "ints": {"x"},
        "optional": set(),
    },
    "cp_async_bulk_prefetch_2d": {
        "required": {"tmap", "x", "y"},
        "ints": {"x", "y"},
        "optional": set(),
    },
    "cp_async_bulk_prefetch_3d": {
        "required": {"tmap", "x", "y", "z"},
        "ints": {"x", "y", "z"},
        "optional": set(),
    },
    "ptx_bar_sync": {
        "required": {"bar_id", "count"},
        "ints": {"bar_id", "count"},
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
    "persistent_loop_begin": {
        "required": set(),
        "optional": {"scheduler"},
    },
    "persistent_loop_end": {
        "required": set(),
        "optional": set(),
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


def _canonical_op_name(kind: str) -> str:
    if kind in OP_ALIASES:
        return OP_ALIASES[kind]
    if kind in KNOWN_OPS:
        return kind
    for prefix, name in TCGEN_PREFIX_CONTRACTS:
        if kind.startswith(prefix):
            return name
    return kind


def _infer_op_metadata(op_name: str, op_args: Dict[str, Any]) -> None:
    if op_name.startswith("tcgen05_ld_"):
        m = re.match(r"tcgen05_ld_(?P<shape>\d+x\d+b)x(?P<num>\d+)", op_name)
        if m:
            op_args.setdefault("shape", m.group("shape"))
            op_args.setdefault("num", int(m.group("num")))
    if op_name.startswith("tcgen05_cp_"):
        m = re.match(r"tcgen05_cp_(?P<shape>\d+x\d+b)(?:_(?P<tile>warpx\d+))?", op_name)
        if m:
            op_args.setdefault("shape", m.group("shape"))
            tile = m.group("tile")
            if tile:
                op_args.setdefault("tile", tile)
    if op_name.startswith("tcgen05_mma_"):
        variant = op_name[len("tcgen05_mma_"):]
        if variant:
            op_args.setdefault("shape", variant.replace("_", "."))
    if op_name.startswith("tma_"):
        if "1d" in op_name:
            op_args.setdefault("rank", 1)
        elif "2d" in op_name:
            op_args.setdefault("rank", 2)
        elif "3d" in op_name:
            op_args.setdefault("rank", 3)
