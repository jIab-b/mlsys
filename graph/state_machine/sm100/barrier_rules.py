from __future__ import annotations

from typing import Any, Dict, Optional

from graph.ir import BarrierState, Graph

from .state import ValidationState


def apply_cluster_sync_flags(canonical: str, state: ValidationState) -> None:
    if canonical == "mbarrier_fence_init_release":
        state.cluster_init_fenced = True
    if canonical == "barrier_cluster_wait":
        state.cluster_sync_done = True


def validate_cluster_barrier_usage(canonical: str, op_name: str, op_args: Dict[str, Any], g: Graph, state: ValidationState) -> None:
    if "bar" not in op_args:
        return
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


def apply_barrier_lifecycle(canonical: str, op_name: str, op_args: Dict[str, Any], state: ValidationState) -> None:
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

    if canonical in ("tma_load", "tma_load_mcast"):
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
