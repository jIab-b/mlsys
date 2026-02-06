from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from graph.core import Graph, Node
from static_validator import _canonical_op_name


def _iter_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_nodes(node.children)


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
            f"(expected one of thread|warp|warpgroup|cta)"
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
