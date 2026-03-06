from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from graph.ir import BarrierState, BufferState, Graph, Node
from graph.ptx_ops.spec import GRAPH_SMEM_LIMIT_BYTES

from .state import ValidationState


def clone_state(state: ValidationState) -> ValidationState:
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


def validate_nodes(
    nodes: List[Node],
    g: Graph,
    state: ValidationState,
    validate_op_fn: Callable[[Node, Graph, ValidationState], None],
) -> None:
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
                raise ValueError(f"KernelStart: smem {total_smem} > assumed limit {GRAPH_SMEM_LIMIT_BYTES} bytes")
            continue

        if node.kind == "KernelEnd":
            for buf, st in state.buf_state.items():
                if st is not None and st != BufferState.EMPTY:
                    raise ValueError(f"Kernel end: buffer {buf} not deallocated ({st})")
            if state.pending_ld is True:
                raise ValueError("Kernel end: pending tcgen05.ld without wait_ld")
            if state.pending_st is True:
                raise ValueError("Kernel end: pending tcgen05.st without wait_st")
            continue

        if node.kind == "Block":
            validate_nodes(node.children, g, state, validate_op_fn)
            continue

        if node.kind == "If":
            if "cond" not in node.args:
                raise ValueError("If node missing 'cond'")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            else_node = next((c for c in node.children if c.kind == "Else"), None)

            s1 = clone_state(state)
            if then_node:
                validate_nodes(then_node.children, g, s1, validate_op_fn)

            s2 = clone_state(state)
            if else_node:
                validate_nodes(else_node.children, g, s2, validate_op_fn)

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

        if node.kind in ("Raw", "Launch"):
            continue

        if node.kind == "Op":
            validate_op_fn(node, g, state)
            continue

        if node.kind == "For":
            if "iters" not in node.args or "var" not in node.args:
                raise ValueError("For node requires 'var' and 'iters'")
            iters = int(node.args["iters"])
            for _ in range(iters):
                validate_nodes(node.children, g, state, validate_op_fn)
            continue

        if node.kind in ("Then", "Else"):
            validate_nodes(node.children, g, state, validate_op_fn)
            continue

        validate_op_fn(node, g, state)
