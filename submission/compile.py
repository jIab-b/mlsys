#!/usr/bin/env python3
"""Build kernel.cu from a DSL graph with combined static + dynamic validation.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; kernel.cu only calls DSL inlines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
OUT_KERNEL = ROOT / "kernel.cu"


class MemSpace(Enum):
    GMEM = auto()
    SMEM = auto()
    TMEM = auto()
    RMEM = auto()


class BarrierState(Enum):
    UNINIT = auto()
    INIT = auto()
    PENDING = auto()
    ARRIVED = auto()
    COMPLETE = auto()


class BufferState(Enum):
    EMPTY = auto()
    FILLING = auto()
    FULL = auto()
    CONSUMING = auto()


@dataclass
class Tensor:
    name: str
    space: MemSpace
    shape: Tuple[int, ...]
    dtype: str


@dataclass
class Barrier:
    name: str
    scope: str  # "cta" or "cluster"


@dataclass
class OpContract:
    name: str
    issue_scope: str  # "one_thread", "one_warp", "all_warps"
    pre: Dict[str, BarrierState] = field(default_factory=dict)
    post: Dict[str, BarrierState] = field(default_factory=dict)
    buffer_pre: Dict[str, BufferState] = field(default_factory=dict)
    buffer_post: Dict[str, BufferState] = field(default_factory=dict)


@dataclass
class Node:
    kind: str
    args: Dict[str, str] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)


class Graph:
    def __init__(self) -> None:
        self.host_nodes: List[Node] = []
        self.device_nodes: List[Node] = []
        self.barriers: Dict[str, Barrier] = {}
        self.buffers: Dict[str, Tensor] = {}

    def add_barrier(self, name: str, scope: str = "cta") -> None:
        self.barriers[name] = Barrier(name=name, scope=scope)

    def add_buffer(self, name: str, space: MemSpace, shape: Tuple[int, ...], dtype: str) -> None:
        self.buffers[name] = Tensor(name=name, space=space, shape=shape, dtype=dtype)

    def add_host_op(self, name: str, **kwargs: str) -> None:
        self.host_nodes.append(Node(kind=name, args=kwargs))

    def add_op(self, name: str, **kwargs: str) -> None:
        self.device_nodes.append(Node(kind=name, args=kwargs))

    def block(self, *nodes: Node) -> Node:
        return Node(kind="Block", children=list(nodes))

    def if_(self, cond: str, then_nodes: List[Node], else_nodes: Optional[List[Node]] = None) -> Node:
        node = Node(kind="If", args={"cond": cond})
        node.children.append(Node(kind="Then", children=then_nodes))
        if else_nodes is not None:
            node.children.append(Node(kind="Else", children=else_nodes))
        return node

    def for_(self, var: str, iters: int, body: List[Node]) -> Node:
        return Node(kind="For", args={"var": var, "iters": str(iters)}, children=body)


# Minimal op contracts (extend as needed)
CONTRACTS: Dict[str, OpContract] = {
    "mbarrier_init": OpContract(
        name="mbarrier_init",
        issue_scope="one_thread",
        pre={},
        post={"bar": BarrierState.INIT},
    ),
    "tma_3d_gmem2smem": OpContract(
        name="tma_3d_gmem2smem",
        issue_scope="one_thread",
        pre={"bar": BarrierState.INIT},
        post={"bar": BarrierState.PENDING},
        buffer_pre={"buf": BufferState.EMPTY},
        buffer_post={"buf": BufferState.FILLING},
    ),
    "mbarrier_arrive_expect_tx": OpContract(
        name="mbarrier_arrive_expect_tx",
        issue_scope="one_thread",
        pre={"bar": BarrierState.PENDING},
        post={"bar": BarrierState.ARRIVED},
        buffer_post={"buf": BufferState.FULL},
    ),
    "mbarrier_wait": OpContract(
        name="mbarrier_wait",
        issue_scope="all_warps",
        pre={"bar": BarrierState.ARRIVED},
        post={"bar": BarrierState.COMPLETE},
    ),
    "tcgen05_cp_scale": OpContract(
        name="tcgen05_cp_scale",
        issue_scope="one_warp",
        buffer_pre={"buf": BufferState.FULL},
    ),
    "tcgen05_mma": OpContract(
        name="tcgen05_mma",
        issue_scope="one_warp",
        buffer_pre={"buf": BufferState.FULL},
    ),
    "tcgen05_commit": OpContract(
        name="tcgen05_commit",
        issue_scope="one_warp",
        pre={"bar": BarrierState.INIT},
        post={"bar": BarrierState.ARRIVED},
    ),
}


def _copy_state(bar_state: Dict[str, BarrierState], buf_state: Dict[str, BufferState]):
    return dict(bar_state), dict(buf_state)


def _validate_op(op: Node, g: Graph, bar_state: Dict[str, BarrierState], buf_state: Dict[str, BufferState]) -> None:
    if op.kind not in CONTRACTS:
        raise ValueError(f"Unknown op: {op.kind}")
    c = CONTRACTS[op.kind]

    for key in list(c.pre.keys()) + list(c.post.keys()):
        if key not in op.args:
            raise ValueError(f"{op.kind}: missing barrier arg '{key}'")
        if op.args[key] not in g.barriers:
            raise ValueError(f"{op.kind}: unknown barrier '{op.args[key]}'")

    for key in list(c.buffer_pre.keys()) + list(c.buffer_post.keys()):
        if key not in op.args:
            raise ValueError(f"{op.kind}: missing buffer arg '{key}'")
        if op.args[key] not in g.buffers:
            raise ValueError(f"{op.kind}: unknown buffer '{op.args[key]}'")

    for key, required in c.pre.items():
        bar = op.args[key]
        if bar_state[bar] != required:
            raise ValueError(f"{op.kind}: barrier {bar} state {bar_state[bar]} != {required}")

    for key, required in c.buffer_pre.items():
        buf = op.args[key]
        if buf_state[buf] != required:
            raise ValueError(f"{op.kind}: buffer {buf} state {buf_state[buf]} != {required}")

    for key, new_state in c.post.items():
        bar = op.args[key]
        bar_state[bar] = new_state

    for key, new_state in c.buffer_post.items():
        buf = op.args[key]
        buf_state[buf] = new_state


def _validate_nodes(nodes: List[Node], g: Graph, bar_state: Dict[str, BarrierState], buf_state: Dict[str, BufferState]) -> None:
    for node in nodes:
        if node.kind == "Block":
            _validate_nodes(node.children, g, bar_state, buf_state)
            continue

        if node.kind == "If":
            if "cond" not in node.args:
                raise ValueError("If node missing 'cond'")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            else_node = next((c for c in node.children if c.kind == "Else"), None)

            b1, f1 = _copy_state(bar_state, buf_state)
            if then_node:
                _validate_nodes(then_node.children, g, b1, f1)

            b2, f2 = _copy_state(bar_state, buf_state)
            if else_node:
                _validate_nodes(else_node.children, g, b2, f2)

            if b1 != b2 or f1 != f2:
                raise ValueError("If branches end in different states; cannot reconcile")

            bar_state.update(b1)
            buf_state.update(f1)
            continue

        if node.kind == "For":
            if "iters" not in node.args or "var" not in node.args:
                raise ValueError("For node requires 'var' and 'iters'")
            iters = int(node.args["iters"])
            for _ in range(iters):
                _validate_nodes(node.children, g, bar_state, buf_state)
            continue

        if node.kind in ("Then", "Else"):
            _validate_nodes(node.children, g, bar_state, buf_state)
            continue

        _validate_op(node, g, bar_state, buf_state)


def validate_graph(g: Graph) -> None:
    bar_state = {name: BarrierState.UNINIT for name in g.barriers}
    buf_state = {name: BufferState.EMPTY for name in g.buffers}
    _validate_nodes(g.device_nodes, g, bar_state, buf_state)


def _emit_nodes(nodes: List[Node], indent: int = 2) -> List[str]:
    lines: List[str] = []
    pad = " " * indent
    for node in nodes:
        if node.kind == "Block":
            lines.extend(_emit_nodes(node.children, indent))
            continue
        if node.kind == "If":
            cond = node.args.get("cond", "/*cond*/")
            lines.append(f"{pad}if ({cond}) {{\n")
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            if then_node:
                lines.extend(_emit_nodes(then_node.children, indent + 2))
            lines.append(f"{pad}}}\n")
            else_node = next((c for c in node.children if c.kind == "Else"), None)
            if else_node:
                lines.append(f"{pad}else {{\n")
                lines.extend(_emit_nodes(else_node.children, indent + 2))
                lines.append(f"{pad}}}\n")
            continue
        if node.kind == "For":
            var = node.args.get("var", "i")
            iters = node.args.get("iters", "0")
            lines.append(f"{pad}for (int {var} = 0; {var} < {iters}; ++{var}) {{\n")
            lines.extend(_emit_nodes(node.children, indent + 2))
            lines.append(f"{pad}}}\n")
            continue
        if node.kind in ("Then", "Else"):
            lines.extend(_emit_nodes(node.children, indent))
            continue

        if "call" in node.args:
            lines.append(f"{pad}{node.args['call']};\n")
        else:
            args = ", ".join(node.args.values())
            lines.append(f"{pad}{node.kind}({args});\n")
    return lines


def emit_kernel(g: Graph, out_path: Path) -> None:
    includes = [
        '#include "ptx_lib/ptx_common.cuh"',
        '#include "ptx_lib/ptx_tma.cuh"',
        '#include "ptx_lib/ptx_mbarrier.cuh"',
        '#include "ptx_lib/ptx_tcgen05_cp.cuh"',
        '#include "ptx_lib/ptx_tcgen05_mma.cuh"',
        '#include "ptx_lib/ptx_tcgen05_ldst.cuh"',
        '#include "ptx_lib/ptx_tcgen05_sync.cuh"',
    ]

    lines: List[str] = []
    lines.append("// AUTO-GENERATED by compile.py\n")
    for inc in includes:
        lines.append(f"{inc}\n")
    lines.append("\n")

    lines.append("extern \"C\" __global__ void kernel_entry() {\n")
    lines.extend(_emit_nodes(g.device_nodes, indent=2))
    lines.append("}\n")

    out_path.write_text("".join(lines))


def build_graph() -> Graph:
    g = Graph()
    # Minimal placeholders to show structure.
    g.add_barrier("tma_bar")
    g.add_buffer("tileA", MemSpace.SMEM, (128, 256), "bf16")

    init_block = g.block(
        Node(kind="mbarrier_init", args={"bar": "tma_bar"}),
    )

    stage_body = [
        Node(kind="tma_3d_gmem2smem", args={"bar": "tma_bar", "buf": "tileA"}),
        Node(kind="mbarrier_arrive_expect_tx", args={"bar": "tma_bar", "buf": "tileA"}),
        Node(kind="mbarrier_wait", args={"bar": "tma_bar"}),
    ]
    stage_loop = g.for_("stage", 1, stage_body)

    g.device_nodes.extend([init_block, stage_loop])
    return g


def main() -> int:
    g = build_graph()
    validate_graph(g)
    emit_kernel(g, OUT_KERNEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
