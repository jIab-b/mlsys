#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from compiler import _write_sub_test, build_gemm1_graph  # noqa: E402
from graph.core import Graph, MemSpace  # noqa: E402
from graph_string import graph_string  # noqa: E402
from parse import _split_python_with_load_inline, _split_with_annotations  # noqa: E402
from state_machine import validate_graph  # noqa: E402

from ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    insert_kernel,
    insert_run,
    reset_ledger,
)

CUDA_EXP_ROOT = ROOT / "cuda_lib" / "experimental"
DEVICE_EXP_DIR = CUDA_EXP_ROOT / "device"
HOST_EXP_DIR = CUDA_EXP_ROOT / "host"
LLM_GRAPHS_DIR = Path(__file__).resolve().parent / "llm_graphs"


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
    source_re = re.compile(r"^(?P<path>[^:]+):(?P<start>\d+)-(?P<end>\d+)(?:\s+chunk=(?P<chunk>\S+))?$")

    for raw in graph_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# sources:"):
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
            raise ValueError(f"{graph_path}: invalid source line: {line}")
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


def _hash_files(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def _ensure_exp_dirs() -> None:
    DEVICE_EXP_DIR.mkdir(parents=True, exist_ok=True)
    HOST_EXP_DIR.mkdir(parents=True, exist_ok=True)
    LLM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def cmd_init_ledger(args: argparse.Namespace) -> int:
    reset_ledger(Path(args.ledger))
    return 0


def cmd_make_graph(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    name = args.name
    graph_path = LLM_GRAPHS_DIR / f"{name}.graph"

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

    device_path = Path(args.device)
    host_path = Path(args.host)
    python_path = Path(args.python)

    sources = {
        "device": _ranges_for(device_path),
        "host": _ranges_for(host_path),
        "python": _ranges_for(python_path),
    }

    g = build_gemm1_graph(
        device_paths=[device_path],
        host_paths=[host_path],
        python_path=python_path,
    )
    validate_graph(g)
    dump = graph_string(g)

    lines = [
        "# sources:device",
        *[
            f"{src.path}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"{src.path}:{src.start}-{src.end}"
            for src in sources["device"]
        ],
        "# sources:host",
        *[
            f"{src.path}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"{src.path}:{src.start}-{src.end}"
            for src in sources["host"]
        ],
        "# sources:python",
        *[
            f"{src.path}:{src.start}-{src.end} chunk={src.chunk}"
            if src.chunk
            else f"{src.path}:{src.start}-{src.end}"
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


def cmd_record_run(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    graph_path = Path(args.graph)
    sources = _parse_sources(graph_path)
    device_files = {s.path for s in sources.get("device", [])}
    host_files = {s.path for s in sources.get("host", [])}
    python_files = {s.path for s in sources.get("python", [])}

    kernel_id = args.kernel_id
    code_hash = _hash_files(sorted(device_files | host_files | python_files))
    graph_hash = _hash_files([graph_path])

    insert_kernel(
        {
            "id": kernel_id,
            "template_id": args.template_id,
            "code_hash": code_hash,
            "graph_hash": graph_hash,
            "cuda_dir": str(CUDA_EXP_ROOT),
            "graph_path": str(graph_path),
            "params": args.params,
            "notes": args.notes,
        },
        path=ledger_path,
    )
    insert_run(
        {
            "id": args.run_id,
            "kernel_id": kernel_id,
            "task_name": args.task,
            "status": args.status,
            "metric": args.metric,
            "time_us": args.time_us,
            "cli_cmd": args.cli_cmd,
            "log_ref": args.log_ref,
        },
        path=ledger_path,
    )
    return 0


def cmd_append_raw(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    section = args.section
    if section not in {"device", "host"}:
        raise ValueError("section must be 'device' or 'host'")
    exp_dir = DEVICE_EXP_DIR if section == "device" else HOST_EXP_DIR
    prefix = "device" if section == "device" else "host"

    existing = sorted(exp_dir.glob(f"{prefix}_*.cuh"))
    if existing:
        target = existing[-1]
    else:
        target = exp_dir / f"{prefix}_00.cuh"

    lines = target.read_text().splitlines() if target.exists() else []
    if len(lines) > 500:
        idx = len(existing)
        target = exp_dir / f"{prefix}_{idx:02d}.cuh"
        lines = []

    with target.open("a", encoding="utf-8") as f:
        if lines and not lines[-1].endswith("\n"):
            f.write("\n")
        f.write(args.code.rstrip() + "\n")
    print(target)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM interface for kernel tooling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-ledger", help="Reset ledger JSON")
    p_init.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    p_init.set_defaults(func=cmd_init_ledger)

    p_graph = sub.add_parser("make-graph", help="Create a graph file from sources")
    p_graph.add_argument("--name", required=True, help="Graph name (llm_graphs/<name>.graph)")
    p_graph.add_argument("--device", default=str(ROOT / "cuda_lib" / "device.cuh"))
    p_graph.add_argument("--host", default=str(ROOT / "cuda_lib" / "host.cuh"))
    p_graph.add_argument("--python", default=str(ROOT / "cuda_lib" / "python.py"))
    p_graph.set_defaults(func=cmd_make_graph)

    p_compile = sub.add_parser("compile", help="Compile from llm graph file")
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

    p_run = sub.add_parser("record-run", help="Record a run in ledger JSON")
    p_run.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    p_run.add_argument("--graph", required=True)
    p_run.add_argument("--kernel-id", required=True)
    p_run.add_argument("--template-id", default=None)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--status", required=True)
    p_run.add_argument("--metric", default=None)
    p_run.add_argument("--time-us", type=float, default=None)
    p_run.add_argument("--cli-cmd", default=None)
    p_run.add_argument("--log-ref", default=None)
    p_run.add_argument("--params", default=None)
    p_run.add_argument("--notes", default=None)
    p_run.set_defaults(func=cmd_record_run)

    p_append = sub.add_parser("append-raw", help="Append raw code to experimental device/host")
    p_append.add_argument("--section", required=True, help="device or host")
    p_append.add_argument("--code", required=True, help="Raw code to append")
    p_append.set_defaults(func=cmd_append_raw)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
