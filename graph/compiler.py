#!/usr/bin/env python3
"""Build sub_test.py from a DSL graph with static validation.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; cuda_lib/* holds
host/device/python raw code. The compiler emits sub_test.py with CUDA_SRC inlined.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CUDA_LIB = REPO_ROOT / "cuda_lib"
PTX_LIB = REPO_ROOT / "ptx_lib"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.core import Graph, MemSpace  # noqa: E402
from emit import _emit_nodes  # noqa: E402
from parse import (  # noqa: E402
    _split_with_annotations,
    _split_python_with_load_inline,
    load_section_nodes,
)
from graph_string import graph_string  # noqa: E402
from state_machine import validate_graph  # noqa: E402

SECTION_FILES = {
    "device": CUDA_LIB / "device.cuh",
    "host": CUDA_LIB / "host.cuh",
    "python": CUDA_LIB / "python.py",
}

# Strip local ptx_lib includes since we inline headers into CUDA_SRC.
_PTX_INCLUDE_PREFIXES = (
    '#include "ptx_lib/',
    '#include "ptx_',
    '#include <ptx_',
    '#include "ptx_common.cuh"',
)


def _strip_ptx_includes(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in _PTX_INCLUDE_PREFIXES):
            continue
        if stripped.startswith("#include") and "ptx_" in stripped and ".cuh" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_ptx_functions(header_text: str) -> set[str]:
    names: set[str] = set()
    for line in header_text.splitlines():
        if "PTX_DEVICE" not in line and "__device__" not in line:
            continue
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            names.add(m.group(1))
    return names


def _select_ptx_headers(ptx_headers: list[Path], sources: list[str]) -> list[Path]:
    combined = "\n".join(sources)
    used_headers: list[Path] = []
    ptx_common = next((h for h in ptx_headers if h.name == "ptx_common.cuh"), None)
    for header in ptx_headers:
        if header.name == "ptx_common.cuh":
            continue
        names = _extract_ptx_functions(header.read_text())
        if any(re.search(rf"\b{re.escape(name)}\s*\(", combined) for name in names):
            used_headers.append(header)
    if used_headers and ptx_common is not None:
        return [ptx_common] + used_headers
    return used_headers if used_headers else ([ptx_common] if ptx_common else [])


def _collect_load_inline_nodes(nodes: List[object]) -> List[object]:
    collected: List[object] = []
    for node in nodes:
        kind = getattr(node, "kind", "")
        if kind == "LoadInline":
            collected.append(node)
        for child in getattr(node, "children", []) or []:
            collected.extend(_collect_load_inline_nodes([child]))
    return collected


def _emit_section_by_name(graph: Graph, section_name: str) -> str:
    return "".join(_emit_nodes(graph.sections.get(section_name, []), indent=0))


def _build_cuda_src(
    header_paths: List[Path],
    device_text: str,
    host_text: str,
) -> str:
    selected_headers = _select_ptx_headers(header_paths, [device_text, host_text])
    parts: List[str] = []
    for header in selected_headers:
        parts.append(f"// ----- {header.name} -----\n")
        header_text = _strip_ptx_includes(header.read_text())
        header_lines = []
        for line in header_text.splitlines():
            if line.strip().startswith("#pragma once"):
                continue
            header_lines.append(line)
        parts.append("\n".join(header_lines))
        if not parts[-1].endswith("\n"):
            parts.append("\n")
        parts.append("\n")

    parts.append("// ----- device.cuh -----\n")
    parts.append(_strip_ptx_includes(device_text))
    if not parts[-1].endswith("\n"):
        parts.append("\n")

    parts.append("// ----- host.cuh -----\n")
    parts.append(_strip_ptx_includes(host_text))
    if not parts[-1].endswith("\n"):
        parts.append("\n")
    return "".join(parts)


def build_gemm1_graph() -> Graph:
    g = Graph()
    missing = [name for name, path in SECTION_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cuda_lib files for gemm1: {missing}")

    load_section_nodes("device", SECTION_FILES["device"], g)
    if not g.buffers:
        g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")
    load_section_nodes("host", SECTION_FILES["host"], g)
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


def _write_sub_test(graph: Graph, out_path: Path) -> None:
    if not PTX_LIB.exists():
        raise FileNotFoundError(f"ptx_lib not found: {PTX_LIB}")

    ptx_headers = sorted(PTX_LIB.glob("*.cuh"))
    if not ptx_headers:
        raise FileNotFoundError(f"No .cuh files found in {PTX_LIB}")

    parts: List[str] = []
    parts.append("# AUTO-GENERATED by run.py\n")
    parts.append("# Do not edit directly; edit cuda_lib/* and ptx_lib/*.cuh instead.\n\n")

    load_inline_nodes = _collect_load_inline_nodes(graph.sections.get("python", []))
    cuda_var_to_sections: dict[str, list[str]] = {}
    for node in load_inline_nodes:
        var = node.args.get("cuda_src_var", "CUDA_SRC")
        sections = node.args.get("sections") or ["device", "host"]
        if var in cuda_var_to_sections and cuda_var_to_sections[var] != sections:
            raise ValueError(f"Conflicting sections for {var}: {cuda_var_to_sections[var]} vs {sections}")
        cuda_var_to_sections[var] = sections
    if not cuda_var_to_sections:
        cuda_var_to_sections["CUDA_SRC"] = ["device", "host"]

    python_src = "".join(_emit_nodes(graph.sections.get("python", []), indent=0))

    for var_name, sections in cuda_var_to_sections.items():
        device_chunks: List[str] = []
        host_chunks: List[str] = []
        for section in sections:
            section_text = _emit_section_by_name(graph, section)
            if section.startswith("device"):
                device_chunks.append(section_text)
            elif section.startswith("host"):
                host_chunks.append(section_text)
            else:
                device_chunks.append(section_text)
        device_src = "".join(device_chunks)
        host_src = "".join(host_chunks)
        cuda_src = _build_cuda_src(ptx_headers, device_src, host_src)
        parts.append(f"{var_name} = r'''\n")
        parts.append(cuda_src)
        parts.append("'''\n\n")

    parts.append(_strip_ptx_includes(python_src))
    if not parts[-1].endswith("\n"):
        parts.append("\n")

    out_path.write_text("".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile graph into sub_test.py")
    parser.add_argument("--dump-graph", action="store_true", help="Print graph structure")
    parser.add_argument("--out", default="sub_test.py", help="Output sub_test.py path (relative to graph)")
    args = parser.parse_args()

    g = build_gemm1_graph()
    validate_graph(g)
    if args.dump_graph:
        print(graph_string(g))

    out_path = (ROOT / args.out).resolve()
    _write_sub_test(g, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
