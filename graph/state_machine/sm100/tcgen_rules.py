from __future__ import annotations

from typing import Any, Dict

from graph.ptx_ops.spec import GRAPH_TMEM_MAX_COLS

from .state import ValidationState


def validate_alloc_dealloc(canonical: str, op_name: str, op_args: Dict[str, Any], state: ValidationState) -> None:
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


def apply_tcgen_pending_rules(canonical: str, op_name: str, state: ValidationState) -> None:
    if canonical in ("tcgen05_cp", "tcgen05_mma"):
        state.pending_tcgen_commit = True

    if canonical == "tcgen05_ld":
        state.pending_ld = True
    if canonical == "tcgen05_st":
        state.pending_st = True

    if canonical == "tcgen05_wait_ld":
        if state.pending_ld is False:
            raise ValueError(f"{op_name}: wait_ld without prior ld")
        state.pending_ld = False if state.pending_ld is True else None

    if canonical == "tcgen05_wait_st":
        if state.pending_st is False:
            raise ValueError(f"{op_name}: wait_st without prior st")
        state.pending_st = False if state.pending_st is True else None

    if canonical == "tcgen05_wait":
        if state.pending_ld is False and state.pending_st is False:
            raise ValueError(f"{op_name}: wait without prior ld/st")
        state.pending_ld = False if state.pending_ld is True else state.pending_ld
        state.pending_st = False if state.pending_st is True else state.pending_st

    if canonical in ("tcgen05_commit", "tcgen05_commit_mcast"):
        if state.pending_tcgen_commit is False:
            raise ValueError(f"{op_name}: commit without prior tcgen05 cp/mma")
