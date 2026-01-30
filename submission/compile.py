#!/usr/bin/env python3
"""Build cuda_lib files from a DSL graph with static validation for tcgen ops.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; cuda_lib/* holds
host/device/python raw code. Non-tcgen code is emitted via Raw nodes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
)

SECTION_FILES = {
    "device": CUDA_LIB / "device.cuh",
    "host": CUDA_LIB / "host.cuh",
    "python": CUDA_LIB / "python.py",
}




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
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_mma": OpContract(
        name="tcgen05_mma",
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_ld": OpContract(
        name="tcgen05_ld",
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_st": OpContract(
        name="tcgen05_st",
        issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL},
    ),
    "tcgen05_commit": OpContract(
        name="tcgen05_commit",
        issue_scope="one_warp",
    ),
    "tcgen05_commit_mcast": OpContract(
        name="tcgen05_commit_mcast",
        issue_scope="one_warp",
    ),
    "tcgen05_wait": OpContract(
        name="tcgen05_wait",
        issue_scope="one_warp",
    ),
    "tcgen05_wait_ld": OpContract(
        name="tcgen05_wait_ld",
        issue_scope="one_warp",
    ),
    "tcgen05_wait_st": OpContract(
        name="tcgen05_wait_st",
        issue_scope="one_warp",
    ),
    "tcgen05_fence": OpContract(
        name="tcgen05_fence",
        issue_scope="one_warp",
    ),
    "tcgen05_fence_before_thread_sync": OpContract(
        name="tcgen05_fence_before_thread_sync",
        issue_scope="one_warp",
    ),
    "tcgen05_fence_after_thread_sync": OpContract(
        name="tcgen05_fence_after_thread_sync",
        issue_scope="one_warp",
    ),
    # mbarrier
    "mbarrier_init": OpContract(name="mbarrier_init", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx": OpContract(name="mbarrier_arrive_expect_tx", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx_cta": OpContract(name="mbarrier_arrive_expect_tx_cta", issue_scope="one_thread"),
    "mbarrier_wait": OpContract(name="mbarrier_wait", issue_scope="all_warps"),
    "mbarrier_wait_ticks": OpContract(name="mbarrier_wait_ticks", issue_scope="all_warps"),
    "mbarrier_wait_relaxed": OpContract(name="mbarrier_wait_relaxed", issue_scope="all_warps"),
    "mbarrier_fence_init_release": OpContract(name="mbarrier_fence_init_release", issue_scope="one_thread"),
    # tma / cp.async bulk
    "tma_gmem2smem": OpContract(name="tma_gmem2smem", issue_scope="one_thread"),
    "tma_1d_gmem2smem": OpContract(name="tma_1d_gmem2smem", issue_scope="one_thread"),
    "tma_2d_gmem2smem": OpContract(name="tma_2d_gmem2smem", issue_scope="one_thread"),
    "tma_3d_gmem2smem": OpContract(name="tma_3d_gmem2smem", issue_scope="one_thread"),
    "tma_1d_gmem2smem_mcast": OpContract(name="tma_1d_gmem2smem_mcast", issue_scope="one_thread"),
    "tma_2d_gmem2smem_mcast": OpContract(name="tma_2d_gmem2smem_mcast", issue_scope="one_thread"),
    "tma_3d_gmem2smem_mcast": OpContract(name="tma_3d_gmem2smem_mcast", issue_scope="one_thread"),
    # ptx common helpers
    "ptx_laneid": OpContract(name="ptx_laneid", issue_scope="one_thread"),
    "ptx_activemask": OpContract(name="ptx_activemask", issue_scope="one_thread"),
    "ptx_elect_one_sync": OpContract(name="ptx_elect_one_sync", issue_scope="one_thread"),
    "ptx_elect_sync": OpContract(name="ptx_elect_sync", issue_scope="one_thread"),
    "ptx_bar_sync": OpContract(name="ptx_bar_sync", issue_scope="all_warps"),
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


def _resolve_contract(kind: str) -> Optional[OpContract]:
    if kind in CONTRACTS:
        return CONTRACTS[kind]
    if kind.startswith("tcgen05_"):
        for prefix, name in TCGEN_PREFIX_CONTRACTS:
            if kind.startswith(prefix):
                return CONTRACTS[name]
    return None


def _copy_state(bar_state: Dict[str, BarrierState], buf_state: Dict[str, BufferState]):
    return dict(bar_state), dict(buf_state)


def _resolve_buffer_arg(op: Node, key: str, g: Graph) -> str:
    if key in op.args:
        return op.args[key]
    if key == "tmem" and g.default_tmem is not None:
        return g.default_tmem
    raise ValueError(f"{op.kind}: missing buffer arg '{key}'")


def _validate_op(op: Node, g: Graph, bar_state: Dict[str, BarrierState], buf_state: Dict[str, BufferState]) -> None:
    c = _resolve_contract(op.kind)
    if c is None:
        raise ValueError(f"Unknown op: {op.kind}")

    resolved_bufs: Dict[str, str] = {}
    for key in list(c.buffer_pre.keys()) + list(c.buffer_post.keys()):
        buf_name = _resolve_buffer_arg(op, key, g)
        if buf_name not in g.buffers:
            raise ValueError(f"{op.kind}: unknown buffer '{buf_name}'")
        resolved_bufs[key] = buf_name

    for key, required in c.pre.items():
        bar = op.args.get(key)
        if bar is None:
            raise ValueError(f"{op.kind}: missing barrier arg '{key}'")
        if bar not in g.barriers:
            raise ValueError(f"{op.kind}: unknown barrier '{bar}'")
        if bar_state[bar] != required:
            raise ValueError(f"{op.kind}: barrier {bar} state {bar_state[bar]} != {required}")

    for key, required in c.buffer_pre.items():
        buf = resolved_bufs[key]
        if buf_state[buf] != required:
            raise ValueError(f"{op.kind}: buffer {buf} state {buf_state[buf]} != {required}")

    for key, new_state in c.post.items():
        bar = op.args[key]
        bar_state[bar] = new_state

    for key, new_state in c.buffer_post.items():
        buf = resolved_bufs[key]
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

        if node.kind in ("Raw", "LoadInline"):
            # Raw nodes are opaque; tcgen validation happens only on explicit nodes.
            continue
        if node.kind == "Event":
            op = node.args.get("op", "")
            if not op:
                raise ValueError("Event node missing 'op'")
            _validate_op(Node(kind=op, args={}), g, bar_state, buf_state)
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
    _validate_nodes(g.sections.get("device", []), g, bar_state, buf_state)


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
        if node.kind == "Event":
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
    lines.append(f"  default_tmem: {g.default_tmem}")
    for section, nodes in g.sections.items():
        lines.append(f"  section:{section} nodes={len(nodes)}")
        for line in _format_node(Node(kind="Block", children=nodes), indent=2):
            lines.append(line)
    return "\n".join(lines)


_TCGEN_CALL_RE = re.compile(r"\b(tcgen05_[A-Za-z0-9_]+)\s*\(")
_LOAD_INLINE_MARKER = "# @@LOAD_INLINE@@"


def _split_device_with_tcgen_events(text: str) -> List[Node]:
    nodes: List[Node] = []
    raw_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            raw_lines.append(line)
            continue
        line_no_comment = line.split("//", 1)[0]
        matches = _TCGEN_CALL_RE.findall(line_no_comment)
        raw_lines.append(line)
        if matches:
            raw = "\n".join(raw_lines)
            if raw:
                nodes.append(Node(kind="Raw", args={"code": raw}))
            raw_lines = []
            for op in matches:
                nodes.append(Node(kind="Event", args={"op": op}))
    if raw_lines:
        nodes.append(Node(kind="Raw", args={"code": "\n".join(raw_lines)}))
    return nodes


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

    g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")
    device_text = SECTION_FILES["device"].read_text()
    for node in _split_device_with_tcgen_events(device_text):
        g.sections["device"].append(node)
    g.add_raw("host", SECTION_FILES["host"].read_text())
    python_text = SECTION_FILES["python"].read_text()
    pre_py, post_py = _split_python_with_load_inline(python_text)
    g.add_raw("python", pre_py)
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
    g.add_raw("python", post_py)
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
