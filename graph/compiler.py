#!/usr/bin/env python3
"""Build cuda_lib files from a DSL graph with static validation for tcgen ops.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; cuda_lib/* holds
host/device/python raw code. Non-tcgen code is emitted via Raw nodes.
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CUDA_LIB = REPO_ROOT / "cuda_lib"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.core import (  # noqa: E402
    BarrierState,
    BufferState,
    Graph,
    MemSpace,
    Node,
    OpContract,
    SourceLoc,
)
from graph.nodes.op import OpNode  # noqa: E402

SECTION_FILES = {
    "device": CUDA_LIB / "device.cuh",
    "host": CUDA_LIB / "host.cuh",
    "python": CUDA_LIB / "python.py",
}

_ANNOT_RE = re.compile(r"^\s*(?://|#)\s*@(?P<kind>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<body>.*)$")
_LOAD_INLINE_MARKER = "# @@LOAD_INLINE@@"

# tcgen op contracts only (everything else must be Raw)
CONTRACTS: Dict[str, OpContract] = {
    # tcgen05 core
    "tcgen05_alloc": OpContract(
        name="tcgen05_alloc",
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.EMPTY},
        buffer_post={"tmem": BufferState.FULL},
    ),
    "tcgen05_dealloc": OpContract(
        name="tcgen05_dealloc",
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL},
        buffer_post={"tmem": BufferState.EMPTY},
    ),
    "tcgen05_cp": OpContract(
        name="tcgen05_cp",
        issue_scope="one_thread",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_mma": OpContract(
        name="tcgen05_mma",
        issue_scope="one_thread",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_ld": OpContract(
        name="tcgen05_ld",
        issue_scope="one_thread",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_st": OpContract(
        name="tcgen05_st",
        issue_scope="one_thread",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_commit": OpContract(
        name="tcgen05_commit",
        issue_scope="one_thread",
    ),
    "tcgen05_commit_mcast": OpContract(
        name="tcgen05_commit_mcast",
        issue_scope="one_thread",
    ),
    "tcgen05_wait": OpContract(
        name="tcgen05_wait",
        issue_scope="one_thread",
    ),
    "tcgen05_wait_ld": OpContract(
        name="tcgen05_wait_ld",
        issue_scope="one_thread",
    ),
    "tcgen05_wait_st": OpContract(
        name="tcgen05_wait_st",
        issue_scope="one_thread",
    ),
    "tcgen05_fence": OpContract(
        name="tcgen05_fence",
        issue_scope="one_thread",
    ),
    "tcgen05_fence_before_thread_sync": OpContract(
        name="tcgen05_fence_before_thread_sync",
        issue_scope="one_thread",
    ),
    "tcgen05_fence_after_thread_sync": OpContract(
        name="tcgen05_fence_after_thread_sync",
        issue_scope="one_thread",
    ),
    # mbarrier
    "mbarrier_init": OpContract(name="mbarrier_init", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx": OpContract(name="mbarrier_arrive_expect_tx", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx_cta": OpContract(name="mbarrier_arrive_expect_tx_cta", issue_scope="one_thread"),
    "mbarrier_wait": OpContract(name="mbarrier_wait", issue_scope="all_warps"),
    "mbarrier_wait_ticks": OpContract(name="mbarrier_wait_ticks", issue_scope="all_warps"),
    "mbarrier_wait_relaxed": OpContract(name="mbarrier_wait_relaxed", issue_scope="all_warps"),
    "mbarrier_fence_init_release": OpContract(name="mbarrier_fence_init_release", issue_scope="one_thread"),
    "barrier_cluster_arrive": OpContract(name="barrier_cluster_arrive", issue_scope="all_warps"),
    "barrier_cluster_wait": OpContract(name="barrier_cluster_wait", issue_scope="all_warps"),
    # tma / cp.async bulk
    "tma_gmem2smem": OpContract(name="tma_gmem2smem", issue_scope="one_thread"),
    "tma_1d_gmem2smem": OpContract(name="tma_1d_gmem2smem", issue_scope="one_thread"),
    "tma_2d_gmem2smem": OpContract(name="tma_2d_gmem2smem", issue_scope="one_thread"),
    "tma_3d_gmem2smem": OpContract(name="tma_3d_gmem2smem", issue_scope="one_thread"),
    "tma_1d_gmem2smem_mcast": OpContract(name="tma_1d_gmem2smem_mcast", issue_scope="one_thread"),
    "tma_2d_gmem2smem_mcast": OpContract(name="tma_2d_gmem2smem_mcast", issue_scope="one_thread"),
    "tma_3d_gmem2smem_mcast": OpContract(name="tma_3d_gmem2smem_mcast", issue_scope="one_thread"),
    "cp_async_bulk_prefetch": OpContract(name="cp_async_bulk_prefetch", issue_scope="one_thread"),
    "cp_async_bulk_prefetch_1d": OpContract(name="cp_async_bulk_prefetch_1d", issue_scope="one_thread"),
    "cp_async_bulk_prefetch_2d": OpContract(name="cp_async_bulk_prefetch_2d", issue_scope="one_thread"),
    "cp_async_bulk_prefetch_3d": OpContract(name="cp_async_bulk_prefetch_3d", issue_scope="one_thread"),
    # ptx common helpers
    "ptx_laneid": OpContract(name="ptx_laneid", issue_scope="one_thread"),
    "ptx_activemask": OpContract(name="ptx_activemask", issue_scope="one_thread"),
    "ptx_elect_one_sync": OpContract(name="ptx_elect_one_sync", issue_scope="one_thread"),
    "ptx_elect_sync": OpContract(name="ptx_elect_sync", issue_scope="one_thread"),
    "ptx_bar_sync": OpContract(name="ptx_bar_sync", issue_scope="all_warps"),
    # host-side metadata ops
    "cute_tmap": OpContract(name="cute_tmap", issue_scope="host"),
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
    "tcgen05_commit": {
        "required": {"bar"},
        "optional": {"cta_group"},
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
}

ISSUE_SCOPES = {"one_thread", "one_warp", "all_warps", "host"}
BARRIER_SCOPES = {"cta", "cluster"}

# Graph assumptions:
# - B200 shared memory per block limit is 227 KiB; keep a 1 KiB safety margin.
# - tmem columns are limited to 512 per CTA.
GRAPH_SMEM_LIMIT_BYTES = 227 * 1024 - 1024
GRAPH_TMEM_MAX_COLS = 512


def _canonical_op_name(kind: str) -> str:
    if kind in OP_ALIASES:
        return OP_ALIASES[kind]
    if kind in CONTRACTS:
        return kind
    for prefix, name in TCGEN_PREFIX_CONTRACTS:
        if kind.startswith(prefix):
            return name
    return kind


def _resolve_contract(kind: str) -> Optional[OpContract]:
    canonical = _canonical_op_name(kind)
    return CONTRACTS.get(canonical)


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
    pending_ld: Optional[bool] = False
    cta_group: Optional[int] = None
    last_alloc_cols: Optional[int] = None


def _parse_value(raw: str) -> Any:
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"0x[0-9a-fA-F]+", raw):
        return int(raw, 16)
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def _parse_kv_tokens(tokens: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Invalid annotation token (expected key=value): {tok}")
        key, value = tok.split("=", 1)
        args[key] = _parse_value(value)
    return args


def _register_buffer(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("@buffer requires name=")
    space = str(args.get("space", "tmem")).lower()
    if space == "tmem":
        mem = MemSpace.TMEM
    elif space == "smem":
        mem = MemSpace.SMEM
    elif space == "gmem":
        mem = MemSpace.GMEM
    else:
        mem = MemSpace.RMEM
    cols = args.get("cols")
    shape = (cols,) if cols is not None else (0,)
    g.add_buffer(str(name), mem, shape, str(args.get("dtype", "opaque")))
    g.buffers[str(name)].meta.update(args)
    if mem == MemSpace.TMEM and isinstance(cols, int) and cols > GRAPH_TMEM_MAX_COLS:
        raise ValueError(f"tmem buffer '{name}' cols {cols} > {GRAPH_TMEM_MAX_COLS}")


def _register_barrier(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("@barrier requires name=")
    scope = str(args.get("scope", "cta"))
    g.add_barrier(str(name), scope=scope)
    g.barriers[str(name)].meta.update(args)


def _register_tmap(g: Graph, args: Dict[str, Any]) -> None:
    name = args.get("name")
    if not name:
        raise ValueError("cute_tmap requires name=")
    g.add_tmap(str(name), args)


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


def _split_with_annotations(text: str, filename: Path, g: Graph) -> List[Node]:
    nodes: List[Node] = []
    raw_lines: List[str] = []
    events: List[Node] = []
    loop_stack: List[Dict[str, Any]] = []
    chunk_name: Optional[str] = None

    def flush_chunk() -> None:
        nonlocal raw_lines, events, chunk_name
        if raw_lines:
            meta = {"chunk": chunk_name} if chunk_name else {}
            nodes.append(Node(kind="Raw", args={"code": "\n".join(raw_lines)}, meta=meta))
        if events:
            nodes.extend(events)
        raw_lines = []
        events = []
        chunk_name = None

    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        m = _ANNOT_RE.match(line)
        if not m:
            raw_lines.append(line)
            continue

        kind = m.group("kind")
        body = m.group("body").strip()
        loc = SourceLoc(filename=str(filename), line=idx, column=1)

        if kind == "chunk":
            flush_chunk()
            raw_lines.append(line)
            args = _parse_kv_tokens(shlex.split(body)) if body else {}
            name = args.get("name") or args.get("id")
            chunk_name = str(name) if name is not None else None
            continue

        # keep annotation line in raw chunk
        raw_lines.append(line)

        if kind == "op":
            tokens = shlex.split(body)
            if not tokens:
                raise ValueError(f"{filename}:{idx}: @op requires op name")
            op_name = tokens[0]
            op_args = _parse_kv_tokens(tokens[1:])
            if loop_stack:
                op_args["_loops"] = [dict(entry) for entry in loop_stack]
            when_cond = op_args.pop("when", None)
            op_node = OpNode(op=op_name, args=op_args, loc=loc)
            if when_cond is not None:
                if_node = Node(kind="If", args={"cond": str(when_cond)}, meta={"validate_only": "true"})
                if_node.children.append(Node(kind="Then", children=[op_node]))
                events.append(if_node)
            else:
                events.append(op_node)
        elif kind == "buffer":
            args = _parse_kv_tokens(shlex.split(body))
            _register_buffer(g, args)
        elif kind == "barrier":
            args = _parse_kv_tokens(shlex.split(body))
            _register_barrier(g, args)
        elif kind == "loop":
            args = _parse_kv_tokens(shlex.split(body))
            if "var" not in args:
                raise ValueError(f"{filename}:{idx}: @loop requires var=")
            loop_stack.append(args)
        elif kind == "endloop":
            if not loop_stack:
                raise ValueError(f"{filename}:{idx}: @endloop without @loop")
            loop_stack.pop()
        elif kind == "kernel":
            args = _parse_kv_tokens(shlex.split(body))
            name = str(args.get("name", "kernel"))
            args["name"] = name
            events.append(Node(kind="KernelStart", args=args, loc=loc))
        elif kind == "endkernel":
            events.append(Node(kind="KernelEnd", args={}, loc=loc))
        else:
            # Unknown annotation: keep it in Raw, but do not validate
            pass

    flush_chunk()
    if loop_stack:
        raise ValueError(f"{filename}: unterminated @loop annotations")
    return nodes


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
    if op.kind == "Event":
        return str(op.args.get("op", "")), {}, op.loc
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

    if canonical == "cute_tmap":
        _register_tmap(g, op_args)
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

    if canonical == "tma_gmem2smem":
        size = op_args.get("size")
        if isinstance(size, int) and size % 16 != 0:
            raise ValueError(f"{op_name}: size {size} must be multiple of 16")
        for key in ("dst_align", "src_align"):
            align = op_args.get(key)
            if isinstance(align, int) and align < 16:
                raise ValueError(f"{op_name}: {key} {align} must be >= 16")

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
        state.pending_ld = True
    if canonical == "tcgen05_wait_ld":
        if state.pending_ld is False:
            raise ValueError(f"{op_name}: wait_ld without prior ld")
        if state.pending_ld is True:
            state.pending_ld = False
        else:
            state.pending_ld = None

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
        pending_ld=state.pending_ld,
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
        pending_ld=_merge_optional(a.pending_ld, b.pending_ld),
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
            state.pending_ld = False
            state.cta_group = None
            state.last_alloc_cols = None
            smem_static = node.args.get("smem_bytes") or node.args.get("smem_static")
            smem_dynamic = node.args.get("smem_dynamic")
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
            state.cta_group = merged.cta_group
            state.last_alloc_cols = merged.last_alloc_cols
            continue

        if node.kind in ("Raw", "LoadInline"):
            # Raw nodes are opaque; validation happens only on annotated ops.
            continue
        if node.kind in ("Op", "Event"):
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
        if node.kind in ("Op", "Event"):
            op_name, op_args, _ = _get_op_info(node)
            canonical = _canonical_op_name(op_name)
            if canonical == "cute_tmap":
                _register_tmap(g, op_args)
        if node.children:
            _collect_tmaps(node.children, g)


def validate_graph(g: Graph) -> None:
    _collect_tmaps(g.sections.get("host", []), g)
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
    )
    _validate_nodes(g.sections.get("device", []), g, state)


def _emit_nodes(
    nodes: List[Node],
    indent: int = 0,
    emit_load_inline: bool = True,
    marker_state: Optional[List[bool]] = None,
) -> List[str]:
    lines: List[str] = []
    pad = " " * indent
    for node in nodes:
        if node.kind == "Block":
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            continue
        if node.kind == "If":
            if node.meta.get("validate_only") == "true":
                continue
            cond = node.args.get("cond", "/*cond*/")
            lines.append(f"{pad}if ({cond}) {{\n")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            if then_node:
                lines.extend(
                    _emit_nodes(
                        then_node.children,
                        indent + 2,
                        emit_load_inline=emit_load_inline,
                        marker_state=marker_state,
                    )
                )
            lines.append(f"{pad}}}\n")
            else_node = next((c for c in node.children if c.kind == "Else"), None)
            if else_node:
                lines.append(f"{pad}else {{\n")
                lines.extend(
                    _emit_nodes(
                        else_node.children,
                        indent + 2,
                        emit_load_inline=emit_load_inline,
                        marker_state=marker_state,
                    )
                )
                lines.append(f"{pad}}}\n")
            continue
        if node.kind == "LoadInline":
            if not emit_load_inline:
                if marker_state is not None:
                    if not marker_state[0]:
                        lines.append(f"{pad}{_LOAD_INLINE_MARKER}\n")
                        marker_state[0] = True
                else:
                    lines.append(f"{pad}{_LOAD_INLINE_MARKER}\n")
                continue
            name = node.args.get("name", "module")
            cuda_src_var = node.args.get("cuda_src_var", "CUDA_SRC")
            cpp_sources = node.args.get("cpp_sources", "")
            extra_cuda_cflags = node.args.get("extra_cuda_cflags", [])
            extra_ldflags = node.args.get("extra_ldflags", [])
            verbose = node.args.get("verbose", False)
            is_python_module = node.args.get("is_python_module", False)
            no_implicit_headers = node.args.get("no_implicit_headers", True)
            lines.append(f"{pad}load_inline(\n")
            lines.append(f"{pad}    {name!r},\n")
            lines.append(f"{pad}    cpp_sources={cpp_sources!r},\n")
            lines.append(f"{pad}    cuda_sources={cuda_src_var},\n")
            lines.append(f"{pad}    verbose={verbose},\n")
            lines.append(f"{pad}    is_python_module={is_python_module},\n")
            lines.append(f"{pad}    no_implicit_headers={no_implicit_headers},\n")
            lines.append(f"{pad}    extra_cuda_cflags={extra_cuda_cflags!r},\n")
            lines.append(f"{pad}    extra_ldflags={extra_ldflags!r},\n")
            lines.append(f"{pad})\n")
            continue
        if node.kind == "Raw":
            raw = node.args.get("code", "")
            if raw:
                lines.append(raw.rstrip("\n") + "\n")
            continue
        if node.kind in ("Event", "Op", "KernelStart", "KernelEnd"):
            continue

        if node.kind == "For":
            var = node.args.get("var", "i")
            iters = node.args.get("iters", "0")
            lines.append(f"{pad}for (int {var} = 0; {var} < {iters}; ++{var}) {{\n")
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent + 2,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            lines.append(f"{pad}}}\n")
            continue
        if node.kind in ("Then", "Else"):
            lines.extend(
                _emit_nodes(
                    node.children,
                    indent,
                    emit_load_inline=emit_load_inline,
                    marker_state=marker_state,
                )
            )
            continue

        if "call" in node.args:
            lines.append(f"{pad}{node.args['call']};\n")
        else:
            args = ", ".join(node.args.values())
            lines.append(f"{pad}{node.kind}({args});\n")
    return lines


def emit_section(
    nodes: List[Node],
    out_path: Path,
    emit_load_inline: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    marker_state = [False] if not emit_load_inline else None
    out_path.write_text(
        "".join(
            _emit_nodes(
                nodes,
                indent=0,
                emit_load_inline=emit_load_inline,
                marker_state=marker_state,
            )
        )
    )


def _format_node(node: Node, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if node.kind == "Raw":
        code = node.args.get("code", "")
        line_count = 0 if not code else code.count("\n") + 1
        meta = "" if not node.meta else f" meta={list(node.meta.keys())}"
        lines = [f"{pad}Raw(lines={line_count}){meta}"]
    elif node.kind == "LoadInline":
        name = node.args.get("name", "module")
        lines = [f"{pad}LoadInline(name={name})"]
    elif node.kind == "Event":
        op = node.args.get("op", "")
        lines = [f"{pad}Event({op})"]
    elif node.kind == "Op":
        op = node.args.get("op", "")
        lines = [f"{pad}Op({op})"]
    elif node.kind == "KernelStart":
        name = node.args.get("name", "")
        lines = [f"{pad}KernelStart({name})"]
    elif node.kind == "KernelEnd":
        lines = [f"{pad}KernelEnd"]
    else:
        args = f" {node.args}" if node.args else ""
        meta = "" if not node.meta else f" meta={list(node.meta.keys())}"
        lines = [f"{pad}{node.kind}{args}{meta}"]
    for child in node.children:
        lines.extend(_format_node(child, indent + 1))
    return lines


def graph_string(g: Graph) -> str:
    lines: List[str] = []
    lines.append("Graph:")
    lines.append(f"  buffers: {list(g.buffers.keys())}")
    lines.append(f"  barriers: {list(g.barriers.keys())}")
    lines.append(f"  tmaps: {list(g.tmaps.keys())}")
    lines.append(f"  default_tmem: {g.default_tmem}")
    for section, nodes in g.sections.items():
        lines.append(f"  section:{section} nodes={len(nodes)}")
        for line in _format_node(Node(kind="Block", children=nodes), indent=2):
            lines.append(line)
    return "\n".join(lines)


def _split_python_with_load_inline(text: str) -> Tuple[str, str]:
    if _LOAD_INLINE_MARKER not in text:
        raise ValueError(f"python.py missing load inline marker: {_LOAD_INLINE_MARKER}")
    pre, post = text.split(_LOAD_INLINE_MARKER, 1)
    return pre.rstrip() + "\n", post.lstrip()


def build_gemm1_graph() -> Graph:
    g = Graph()
    missing = [name for name, path in SECTION_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cuda_lib files for gemm1: {missing}")

    device_text = SECTION_FILES["device"].read_text()
    for node in _split_with_annotations(device_text, SECTION_FILES["device"], g):
        g.sections["device"].append(node)
    if not g.buffers:
        g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")
    host_text = SECTION_FILES["host"].read_text()
    for node in _split_with_annotations(host_text, SECTION_FILES["host"], g):
        g.sections["host"].append(node)
    python_text = SECTION_FILES["python"].read_text()
    pre_py, post_py = _split_python_with_load_inline(python_text)
    for node in _split_with_annotations(pre_py, SECTION_FILES["python"], g):
        g.sections["python"].append(node)
    g.add_load_inline(
        "python",
        name="gemm_all",
        cuda_src_var="CUDA_SRC",
        cpp_sources="",
        verbose=False,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        extra_ldflags=["-lcuda"],
    )
    for node in _split_with_annotations(post_py, SECTION_FILES["python"], g):
        g.sections["python"].append(node)
    return g


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile graph into cuda_lib/*")
    parser.add_argument("--dump-graph", action="store_true", help="Print graph structure")
    parser.add_argument("--no-emit", action="store_true", help="Skip writing cuda_lib files")
    args = parser.parse_args()

    g = build_gemm1_graph()
    validate_graph(g)
    if args.dump_graph:
        print(graph_string(g))
    if not args.no_emit:
        for section, path in SECTION_FILES.items():
            emit_section(
                g.sections.get(section, []),
                path,
                emit_load_inline=(section != "python"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
