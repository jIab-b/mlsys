"""SM100 dynamic graph validator orchestrator."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from graph.ir import BarrierState, BufferState, Graph, MemSpace, Node, OpContract, SourceLoc
from graph.ptx_ops.contracts import CONTRACTS
from graph.ptx_ops.spec import (
    BARRIER_SCOPES,
    GRAPH_SMEM_LIMIT_BYTES,
    ISSUE_SCOPES,
    OP_ARG_SPECS,
    _canonical_op_name,
)
from graph.ptx_ops.static import validate_ptx_op

from .barrier_rules import apply_barrier_lifecycle, apply_cluster_sync_flags, validate_cluster_barrier_usage
from .control_flow import validate_nodes
from .idesc import validate_tcgen_descriptor_layout
from .protocol_rules import validate_graph_protocol
from .state import ValidationState
from .tcgen_rules import apply_tcgen_pending_rules, validate_alloc_dealloc
from .tma_rules import validate_tma_op


def _resolve_contract(kind: str) -> Optional[OpContract]:
    return CONTRACTS.get(_canonical_op_name(kind))


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
    contract = _resolve_contract(canonical)
    if contract is None:
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
    if issue_val is not None and issue_val != contract.issue_scope:
        raise ValueError(f"{op_name}: issue_scope {issue_val} != {contract.issue_scope}")

    if isinstance(scope_val, str) and scope_val in BARRIER_SCOPES and "bar" in op_args:
        bar_name = op_args["bar"]
        if bar_name in g.barriers and g.barriers[bar_name].scope != scope_val:
            raise ValueError(f"{op_name}: barrier '{bar_name}' scope {g.barriers[bar_name].scope} != {scope_val}")

    if canonical.startswith("tcgen05_"):
        cta_group = op_args.get("cta_group", 1)
        if isinstance(cta_group, int) and cta_group not in (1, 2):
            raise ValueError(f"{op_name}: invalid cta_group {cta_group}")
        if state.cta_group is None:
            state.cta_group = cta_group if isinstance(cta_group, int) else None
        elif isinstance(cta_group, int) and state.cta_group is not None and cta_group != state.cta_group:
            raise ValueError(f"{op_name}: cta_group {cta_group} != {state.cta_group}")

    resolved_bufs: Dict[str, str] = {}
    for key in list(contract.buffer_pre.keys()) + list(contract.buffer_post.keys()):
        buf_name = _resolve_buffer_arg(op, key, g)
        if buf_name not in g.buffers:
            raise ValueError(f"{op_name}: unknown buffer '{buf_name}'")
        resolved_bufs[key] = buf_name

    if "bar" in op_args:
        bar = op_args["bar"]
        if bar not in g.barriers:
            raise ValueError(f"{op_name}: unknown barrier '{bar}'")
        if canonical != "mbarrier_init" and state.bar_state.get(bar) == BarrierState.UNINIT:
            raise ValueError(f"{op_name}: barrier '{bar}' used before init")

    for key, required in contract.pre.items():
        bar = op_args.get(key)
        if bar is None:
            raise ValueError(f"{op_name}: missing barrier arg '{key}'")
        if bar not in g.barriers:
            raise ValueError(f"{op_name}: unknown barrier '{bar}'")
        if state.bar_state[bar] is not None and state.bar_state[bar] != required:
            raise ValueError(f"{op_name}: barrier {bar} state {state.bar_state[bar]} != {required}")

    for key, required in contract.buffer_pre.items():
        buf = resolved_bufs[key]
        if state.buf_state[buf] is not None and state.buf_state[buf] != required:
            raise ValueError(f"{op_name}: buffer {buf} state {state.buf_state[buf]} != {required}")

    validate_alloc_dealloc(canonical, op_name, op_args, state)
    validate_tcgen_descriptor_layout(canonical, op_name, op_args, g)
    validate_tma_op(canonical, op_name, op_args, g, state)
    apply_tcgen_pending_rules(canonical, op_name, state)
    apply_cluster_sync_flags(canonical, state)
    validate_cluster_barrier_usage(canonical, op_name, op_args, g, state)
    apply_barrier_lifecycle(canonical, op_name, op_args, state)

    for key, new_state in contract.post.items():
        bar = op_args[key]
        state.bar_state[bar] = new_state

    for key, new_state in contract.buffer_post.items():
        buf = resolved_bufs[key]
        state.buf_state[buf] = new_state


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
    if key in {"f16", "half", "fp16", "bf16", "bfloat16", "i16", "int16", "u16", "uint16"}:
        return 2
    if key in {"f32", "float", "fp32", "i32", "int32", "u32", "uint32", "int"}:
        return 4
    if key in {"f64", "double", "fp64", "i64", "int64", "u64", "uint64"}:
        return 8
    if key in {"i8", "int8", "u8", "uint8"}:
        return 1
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
        if size is None or not buf.shape or any(not isinstance(dim, int) for dim in buf.shape):
            return None
        elems = 1
        for dim in buf.shape:
            elems *= dim
        total += elems * size
    return total


def _iter_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_nodes(node.children)


def _validate_graph_ptx_spec(g: Graph) -> None:
    for node in _iter_nodes(g.nodes):
        if node.kind != "Op":
            continue
        op_name = str(node.args.get("op", ""))
        op_args = dict(node.args.get("op_args", {}))
        validate_ptx_op(op_name, op_args)


def validate_graph(g: Graph) -> None:
    """Run all SM100 validation passes on a Graph."""
    _collect_tmaps(g.nodes, g)
    _validate_graph_ptx_spec(g)
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

    validate_nodes(g.nodes, g, state, _validate_op)
    validate_graph_protocol(g, strict=bool(g.meta.get("strict_protocol", False)))
