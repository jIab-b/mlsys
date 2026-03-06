from __future__ import annotations

from typing import Iterable, Optional

from graph.ir import Graph, Node
from graph.ptx_ops.spec import _canonical_op_name

from .state import KernelProtocolState


def _iter_op_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_op_nodes(node.children)


def _proto_add_optional(cur: Optional[int], delta: Optional[int]) -> Optional[int]:
    if cur is None or delta is None:
        return None
    return cur + delta


def validate_graph_protocol(g: Graph, *, strict: bool = False) -> None:
    state = KernelProtocolState()

    for node in _iter_op_nodes(g.nodes):
        if node.kind == "KernelStart":
            state = KernelProtocolState(active=True, name=str(node.args.get("name", "")))
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
            state.bar_expected[bar] = _proto_add_optional(state.bar_expected.get(bar, 0), size_int)
            if bar not in state.bar_completed:
                state.bar_completed[bar] = 0

        if canonical in ("tma_load", "tma_load_mcast"):
            bar = str(op_args.get("bar", ""))
            size = op_args.get("size")
            size_int = int(size) if isinstance(size, int) else None
            state.bar_completed[bar] = _proto_add_optional(state.bar_completed.get(bar, 0), size_int)
            if bar not in state.bar_expected:
                state.bar_expected[bar] = 0

        if canonical in {"mbarrier_wait", "mbarrier_wait_relaxed", "mbarrier_wait_ticks"}:
            bar = str(op_args.get("bar", ""))
            expected = state.bar_expected.get(bar)
            completed = state.bar_completed.get(bar)
            if expected is not None and completed is not None and completed > expected:
                raise ValueError(f"{op_name}: barrier '{bar}' completed bytes {completed} > expected {expected}")
            state.bar_expected[bar] = 0
            state.bar_completed[bar] = 0

        if canonical == "tma_store":
            tmap_name = op_args.get("tmap")
            if tmap_name is not None and tmap_name not in g.tmaps:
                raise ValueError(f"{op_name}: unknown output tmap '{tmap_name}'")
