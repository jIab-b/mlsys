#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = ROOT / "graph"
if str(GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPH_ROOT))

from compiler import _write_sub_test  # noqa: E402
from graph.core import Graph, MemSpace  # noqa: E402
from graph_string import graph_string  # noqa: E402
from parse import _split_python_with_load_inline, _split_with_annotations  # noqa: E402
from state_machine import validate_graph  # noqa: E402

CUDA_EXP_ROOT = ROOT / "cuda_lib" / "experimental"
DEVICE_EXP_DIR = CUDA_EXP_ROOT / "device"
HOST_EXP_DIR = CUDA_EXP_ROOT / "host"
PY_EXP_DIR = CUDA_EXP_ROOT / "python"
KERNEL_GRAPHS_DIR = GRAPH_ROOT / "kernel_graphs"


@dataclass
class SourceRange:
    path: Path
    start: int
    end: int
    chunk: Optional[str] = None


def _read_range(path: Path, start: int, end: int) -> str:
    lines = path.read_text().splitlines()
    slice_lines = lines[start - 1 : end]
    text = "\n".join(slice_lines)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _parse_sources(graph_path: Path) -> Dict[str, List[SourceRange]]:
    sources: Dict[str, List[SourceRange]] = {"device": [], "host": [], "python": []}
    section: Optional[str] = None
    source_re = re.compile(r"^Raw\s+(?P<path>[^:]+):(?P<start>\d+)-(?P<end>\d+)(?:\s+chunk=(?P<chunk>\S+))?$")

    for raw in graph_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# section:"):
            section = line.split(":", 1)[1].strip()
            if section not in sources:
                sources[section] = []
            continue
        if line.startswith("# graph:"):
            break
        if line.startswith("#"):
            continue
        if section is None:
            continue
        m = source_re.match(line)
        if not m:
            raise ValueError(f"{graph_path}: invalid raw line: {line}")
        path = (ROOT / m.group("path")).resolve()
        sources[section].append(
            SourceRange(
                path=path,
                start=int(m.group("start")),
                end=int(m.group("end")),
                chunk=m.group("chunk"),
            )
        )
    return sources


def _build_graph_from_sources(sources: Dict[str, List[SourceRange]]) -> Graph:
    g = Graph()

    for entry in sources.get("device", []):
        text = _read_range(entry.path, entry.start, entry.end)
        nodes = _split_with_annotations(
            text,
            entry.path,
            g,
            line_offset=entry.start - 1,
            initial_chunk=entry.chunk,
            allow_chunk=False,
        )
        g.sections["device"].extend(nodes)
    if not g.buffers:
        g.add_buffer("tmem0", MemSpace.TMEM, (0,), "opaque")

    for entry in sources.get("host", []):
        text = _read_range(entry.path, entry.start, entry.end)
        nodes = _split_with_annotations(
            text,
            entry.path,
            g,
            line_offset=entry.start - 1,
            initial_chunk=entry.chunk,
            allow_chunk=False,
        )
        g.sections["host"].extend(nodes)

    python_parts = []
    for entry in sources.get("python", []):
        python_parts.append(_read_range(entry.path, entry.start, entry.end))
    python_text = "".join(python_parts)
    pre_py, post_py = _split_python_with_load_inline(python_text)
    for node in _split_with_annotations(pre_py, Path("python.py"), g):
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
    for node in _split_with_annotations(post_py, Path("python.py"), g):
        g.sections["python"].append(node)
    return g


def _ensure_exp_dirs() -> None:
    DEVICE_EXP_DIR.mkdir(parents=True, exist_ok=True)
    HOST_EXP_DIR.mkdir(parents=True, exist_ok=True)
    PY_EXP_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def cmd_make_graph(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    name = args.name
    graph_path = KERNEL_GRAPHS_DIR / f"{name}.graph"

    def _ranges_for(path: Path) -> List[SourceRange]:
        text = path.read_text()
        from parse import _parse_chunked_source  # local import to keep dependencies light

        try:
            chunked = _parse_chunked_source(text, path)
        except ValueError:
            chunked = None
        if not chunked:
            lines = text.splitlines()
            if not lines:
                return []
            return [SourceRange(path=path, start=1, end=len(lines), chunk=None)]
        ranges: List[SourceRange] = []
        for name, chunk_text, start_line in chunked:
            line_count = len(chunk_text.splitlines()) if chunk_text else 0
            end_line = start_line + line_count - 1 if line_count > 0 else start_line
            ranges.append(SourceRange(path=path, start=start_line, end=end_line, chunk=name))
        return ranges

    def _ranges_for_dir(dir_path: Path, prefix: str) -> List[SourceRange]:
        ranges: List[SourceRange] = []
        for path in sorted(dir_path.glob(f"{prefix}_*.cuh")):
            ranges.extend(_ranges_for(path))
        return ranges

    device_path = Path(args.device)
    host_path = Path(args.host)
    python_path = Path(args.python)

    exp_device = _ranges_for_dir(DEVICE_EXP_DIR, "device")
    exp_host = _ranges_for_dir(HOST_EXP_DIR, "host")

    sources = {
        "device": _ranges_for(device_path) + exp_device,
        "host": _ranges_for(host_path) + exp_host,
        "python": _ranges_for(python_path),
    }

    g = _build_graph_from_sources(sources)
    validate_graph(g)
    dump = graph_string(g)

    def _fmt_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    lines = [
        "# section:device",
        *[
            f"Raw {_fmt_path(src.path)}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"Raw {_fmt_path(src.path)}:{src.start}-{src.end}"
            for src in sources["device"]
        ],
        "# section:host",
        *[
            f"Raw {_fmt_path(src.path)}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"Raw {_fmt_path(src.path)}:{src.start}-{src.end}"
            for src in sources["host"]
        ],
        "# section:python",
        *[
            f"Raw {_fmt_path(src.path)}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"Raw {_fmt_path(src.path)}:{src.start}-{src.end}"
            for src in sources["python"]
        ],
        "# graph:",
        dump,
        "",
    ]
    graph_path.write_text("\n".join(lines))
    print(graph_path)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    graph_path = Path(args.graph)
    sources = _parse_sources(graph_path)
    g = _build_graph_from_sources(sources)
    validate_graph(g)
    if args.dump_graph:
        print(graph_string(g))
    out_path = (GRAPH_ROOT / args.out).resolve()
    _write_sub_test(g, out_path)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "eval_test" / "modal" / "cli.py"),
        "-t",
        args.task,
        "-m",
        args.mode,
        args.submission,
        "-o",
        args.out,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return result.returncode


def cmd_append_raw(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    section = args.section
    if section not in {"device", "host", "python"}:
        raise ValueError("section must be 'device', 'host', or 'python'")
    if section == "device":
        exp_dir = DEVICE_EXP_DIR
        prefix = "device"
        suffix = ".cuh"
    elif section == "host":
        exp_dir = HOST_EXP_DIR
        prefix = "host"
        suffix = ".cuh"
    else:
        exp_dir = PY_EXP_DIR
        prefix = "python"
        suffix = ".py"

    existing = sorted(exp_dir.glob(f"{prefix}_*{suffix}"))
    default_target = exp_dir / f"{prefix}_00{suffix}"
    target = Path(args.path).resolve() if args.path else (existing[-1] if existing else default_target)
    if target.parent.resolve() != exp_dir.resolve():
        raise ValueError(f"path must be inside {exp_dir}")
    if not target.exists() and args.path:
        target.touch()

    lines = target.read_text().splitlines() if target.exists() else []
    if len(lines) >= 500:
        idx = len(existing)
        target = exp_dir / f"{prefix}_{idx:02d}{suffix}"
        lines = []

    expected_start = len(lines) + 1
    if args.start_line is not None and args.start_line != expected_start:
        raise ValueError(f"{target}: start_line {args.start_line} != expected {expected_start}")

    code_lines = args.code.rstrip().splitlines() if args.code else [""]
    with target.open("a", encoding="utf-8") as f:
        if lines:
            f.write("\n")
        f.write(args.code.rstrip() + "\n")
    start = expected_start
    end = expected_start + len(code_lines) - 1
    rel_target = target.relative_to(ROOT)
    print(f"{rel_target}:{start}-{end}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM interface for kernel tooling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_graph = sub.add_parser("make-graph", help="Create a graph file from sources")
    p_graph.add_argument("--name", required=True, help="Graph name (kernel_graphs/<name>.graph)")
    p_graph.add_argument("--device", default=str(ROOT / "cuda_lib" / "device.cuh"))
    p_graph.add_argument("--host", default=str(ROOT / "cuda_lib" / "host.cuh"))
    p_graph.add_argument("--python", default=str(ROOT / "cuda_lib" / "python.py"))
    p_graph.set_defaults(func=cmd_make_graph)

    p_compile = sub.add_parser("compile", help="Compile from graph file")
    p_compile.add_argument("--graph", required=True, help="Path to .graph file")
    p_compile.add_argument("--out", default="sub_test.py")
    p_compile.add_argument("--dump-graph", action="store_true")
    p_compile.set_defaults(func=cmd_compile)

    p_eval = sub.add_parser("eval", help="Run eval CLI")
    p_eval.add_argument("--task", default="gemm")
    p_eval.add_argument("--mode", default="benchmark")
    p_eval.add_argument("--submission", default="graph/sub_test.py")
    p_eval.add_argument("--out", default="out_test.txt")
    p_eval.set_defaults(func=cmd_eval)

    p_append = sub.add_parser("append-raw", help="Append raw code to experimental device/host")
    p_append.add_argument("--section", required=True, help="device or host")
    p_append.add_argument("--code", required=True, help="Raw code to append")
    p_append.add_argument("--path", default=None, help="Explicit experimental file path to append to")
    p_append.add_argument("--start-line", type=int, default=None, help="Expected start line for append validation")
    p_append.set_defaults(func=cmd_append_raw)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
