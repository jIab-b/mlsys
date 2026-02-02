#!/usr/bin/env python3
"""Build sub_test.py from a DSL graph with static validation.

The graph is the source of truth. PTX stays in ptx_lib/*.cuh; cuda_lib/* holds
host/device/python raw code. The compiler emits sub_test.py with CUDA_SRC inlined.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import re
import sys
from pathlib import Path
from typing import List, Optional

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


def _expand_inputs(values: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for value in values:
        pattern = value
        if not Path(value).is_absolute():
            pattern = str(REPO_ROOT / value)
        matches = [Path(p).resolve() for p in glob.glob(pattern, recursive=True)]
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(Path(pattern).resolve())
    return expanded


def _load_graph_py(path: Path, fn_name: str = "build_graph") -> Graph:
    spec = importlib.util.spec_from_file_location("graph_module", str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load graph module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"{path}: missing function '{fn_name}'")
    graph = fn()
    if not isinstance(graph, Graph):
        raise TypeError(f"{path}: '{fn_name}' did not return Graph")
    return graph


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


def build_gemm1_graph(
    device_paths: Optional[List[Path]] = None,
    host_paths: Optional[List[Path]] = None,
    python_path: Optional[Path] = None,
) -> Graph:
    g = Graph()
    device_paths = device_paths or [SECTION_FILES["device"]]
    host_paths = host_paths or [SECTION_FILES["host"]]
    python_path = python_path or SECTION_FILES["python"]

    missing: list[str] = []
    for path in device_paths + host_paths + [python_path]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    for path in device_paths:
        load_section_nodes("device", path, g)
    if not g.buffers:
        g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")
    for path in host_paths:
        load_section_nodes("host", path, g)
    python_text = python_path.read_text()
    pre_py, post_py = _split_python_with_load_inline(python_text)
    for node in _split_with_annotations(pre_py, python_path, g):
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
    for node in _split_with_annotations(post_py, python_path, g):
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
    parser.add_argument(
        "--graph-py",
        default=None,
        help="Load graph from a Python file (expects build_graph()). Path is relative to repo root.",
    )
    parser.add_argument(
        "--graph-fn",
        default="build_graph",
        help="Function name in --graph-py that returns a Graph (default: build_graph).",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        help="Device source file(s). If set, overrides default cuda_lib/device.cuh. Paths are relative to repo root.",
    )
    parser.add_argument(
        "--host",
        action="append",
        default=[],
        help="Host source file(s). If set, overrides default cuda_lib/host.cuh. Paths are relative to repo root.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python wrapper file. Defaults to cuda_lib/python.py. Path is relative to repo root.",
    )
    parser.add_argument(
        "--extra-device",
        action="append",
        default=[],
        help="Additional device source file(s) appended after --device/default.",
    )
    parser.add_argument(
        "--extra-host",
        action="append",
        default=[],
        help="Additional host source file(s) appended after --host/default.",
    )
    args = parser.parse_args()

    if args.graph_py:
        graph_paths = _expand_inputs([args.graph_py])
        if len(graph_paths) != 1:
            raise ValueError(f"--graph-py must resolve to a single file, got {graph_paths}")
        g = _load_graph_py(graph_paths[0], args.graph_fn)
    else:
        device_paths = _expand_inputs(args.device) if args.device else [SECTION_FILES["device"]]
        host_paths = _expand_inputs(args.host) if args.host else [SECTION_FILES["host"]]
        if args.extra_device:
            device_paths.extend(_expand_inputs(args.extra_device))
        if args.extra_host:
            host_paths.extend(_expand_inputs(args.extra_host))
        python_paths = _expand_inputs([args.python]) if args.python else [SECTION_FILES["python"]]
        if len(python_paths) != 1:
            raise ValueError(f"--python must resolve to a single file, got {python_paths}")
        python_path = python_paths[0]

        g = build_gemm1_graph(
            device_paths=device_paths,
            host_paths=host_paths,
            python_path=python_path,
        )
    validate_graph(g)
    if args.dump_graph:
        print(graph_string(g))

    out_path = (ROOT / args.out).resolve()
    _write_sub_test(g, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
