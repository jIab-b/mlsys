from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from graph.core import Graph, Node
from ptx_ops.spec import (
    PTX_TCGEN05_CP_SHAPE_TILE,
    PTX_TCGEN05_MMA_SHAPES,
    PTX_TCGEN05_NO_TRANSPOSE_KINDS,
    TMA_INTERLEAVE_SET,
    TMA_SWIZZLE_SET,
    _canonical_op_name,
)


def _iter_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_nodes(node.children)


# -------------------------
# PTX-form validation
# -------------------------


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


# -------------------------
# Protocol validation
# -------------------------


@dataclass
class _KernelProtocolState:
    active: bool = False
    name: str = ""
    bar_expected: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_completed: Dict[str, Optional[int]] = field(default_factory=dict)
    saw_group2: bool = False
    saw_group2_marker: bool = False


def _add_optional(cur: Optional[int], delta: Optional[int]) -> Optional[int]:
    if cur is None or delta is None:
        return None
    return cur + delta


def _normalize_exec_scope(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _normalize_issuer(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _validate_exec_contract(op_name: str, op_args: Dict[str, object]) -> None:
    exec_scope = _normalize_exec_scope(op_args.get("exec_scope"))
    issuer = _normalize_issuer(op_args.get("issuer"))
    if exec_scope not in {"thread", "warp", "warpgroup", "cta"}:
        raise ValueError(
            f"{op_name}: missing/invalid exec_scope "
            "(expected one of thread|warp|warpgroup|cta)"
        )
    if issuer not in {"all", "single", "mask"}:
        raise ValueError(f"{op_name}: missing/invalid issuer (expected all|single|mask)")
    if issuer == "single" and "lane_pred" not in op_args:
        raise ValueError(f"{op_name}: issuer=single requires lane_pred metadata")


def validate_graph_protocol(g: Graph, *, strict: bool = False) -> None:
    state = _KernelProtocolState()

    for node in _iter_nodes(g.sections.get("device", [])):
        if node.kind == "KernelStart":
            state = _KernelProtocolState(active=True, name=str(node.args.get("name", "")))
            continue

        if node.kind == "KernelEnd":
            if not state.active:
                continue
            for bar, expected in state.bar_expected.items():
                completed = state.bar_completed.get(bar)
                if expected is None or completed is None:
                    continue
                if completed > expected:
                    raise ValueError(
                        f"kernel {state.name}: barrier '{bar}' completed bytes {completed} > expected {expected}"
                    )
                if expected != 0 or completed != 0:
                    raise ValueError(
                        f"kernel {state.name}: barrier '{bar}' left with expected={expected} completed={completed} "
                        "(mbarrier lifecycle not closed)"
                    )
            if state.saw_group2 and strict and not state.saw_group2_marker:
                raise ValueError(
                    f"kernel {state.name}: saw cta_group=2 instructions without explicit cta_group_set marker"
                )
            state.active = False
            continue

        if node.kind != "Op":
            continue

        op_name = str(node.args.get("op", ""))
        op_args = dict(node.args.get("op_args", {}))
        canonical = _canonical_op_name(op_name)

        if strict and (
            canonical.startswith("tcgen05_")
            or canonical.startswith("tma_")
            or canonical.startswith("mbarrier_")
            or canonical.startswith("barrier_cluster_")
            or canonical == "ptx_bar_sync"
            or canonical == "tmap_create"
            or canonical == "tma_store_out"
        ):
            _validate_exec_contract(op_name, op_args)

        if canonical == "cta_group_set":
            if op_args.get("value") == 2:
                state.saw_group2_marker = True
            continue

        cta_group = op_args.get("cta_group")
        if cta_group == 2:
            state.saw_group2 = True

        if canonical in {"mbarrier_arrive_expect_tx", "mbarrier_arrive_expect_tx_cta"}:
            bar = str(op_args.get("bar", ""))
            size = op_args.get("size")
            size_int = int(size) if isinstance(size, int) else None
            state.bar_expected[bar] = _add_optional(state.bar_expected.get(bar, 0), size_int)
            if bar not in state.bar_completed:
                state.bar_completed[bar] = 0

        if canonical.startswith("tma_") and canonical.endswith("gmem2smem"):
            bar = str(op_args.get("bar", ""))
            size = op_args.get("size")
            size_int = int(size) if isinstance(size, int) else None
            state.bar_completed[bar] = _add_optional(state.bar_completed.get(bar, 0), size_int)
            if bar not in state.bar_expected:
                state.bar_expected[bar] = 0

        if canonical in {"mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"}:
            bar = str(op_args.get("bar", ""))
            expected = state.bar_expected.get(bar)
            completed = state.bar_completed.get(bar)
            if expected is not None and completed is not None and completed > expected:
                raise ValueError(
                    f"{op_name}: barrier '{bar}' completed bytes {completed} > expected {expected}"
                )
            state.bar_expected[bar] = 0
            state.bar_completed[bar] = 0

        if canonical in {"tma_store_out", "tma_1d_smem2gmem", "tma_2d_smem2gmem", "tma_3d_smem2gmem"}:
            tmap_name = op_args.get("tmap")
            if tmap_name is not None and tmap_name not in g.tmaps:
                raise ValueError(f"{op_name}: unknown output tmap '{tmap_name}'")
