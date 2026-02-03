#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import re
import shutil
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
BACKUP_DIR = ROOT / ".llm_backups"


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
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    _ensure_exp_dirs()
    rel = path.resolve()
    try:
        rel = rel.relative_to(ROOT)
        rel_str = str(rel).replace("/", "__")
    except ValueError:
        rel_str = rel.name
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_path = BACKUP_DIR / f"{stamp}__{rel_str}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def _normalize_code(code: str | None) -> list[str]:
    if not code:
        return []
    text = code
    if text.endswith("\n"):
        text = text[:-1]
    return text.splitlines()


def _apply_edit_range(
    path: Path,
    start: int,
    end: int,
    code: str | None,
    mode: str,
) -> tuple[int, int]:
    lines = path.read_text().splitlines() if path.exists() else []
    total = len(lines)

    if mode == "insert_before":
        if start < 1 or start > total + 1:
            raise ValueError(f"{path}: start_line out of range for insert_before")
        new_lines = _normalize_code(code)
        idx = start - 1
        lines = lines[:idx] + new_lines + lines[idx:]
        start_out = start
        end_out = start + len(new_lines) - 1
    elif mode == "insert_after":
        if start < 1 or start > total:
            raise ValueError(f"{path}: start_line out of range for insert_after")
        new_lines = _normalize_code(code)
        idx = start
        lines = lines[:idx] + new_lines + lines[idx:]
        start_out = start + 1
        end_out = start + len(new_lines)
    else:
        if start < 1 or end < start or end > total:
            raise ValueError(f"{path}: invalid start/end for replace/delete")
        new_lines = [] if mode == "delete" else _normalize_code(code)
        lines = lines[: start - 1] + new_lines + lines[end:]
        start_out = start
        end_out = start + len(new_lines) - 1

    text = "\n".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text)
    return start_out, end_out


def _find_matching_brace(text: str, start_idx: int) -> int:
    depth = 0
    i = start_idx
    in_str: str | None = None
    triple = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if triple:
                if text.startswith(in_str * 3, i):
                    i += 3
                    in_str = None
                    triple = False
                    continue
            else:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    i += 1
                    in_str = None
                    continue
            i += 1
            continue
        if ch in ("'", '"'):
            if text.startswith(ch * 3, i):
                in_str = ch
                triple = True
                i += 3
                continue
            in_str = ch
            triple = False
            i += 1
            continue
        if ch == "#":
            nl = text.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("matching brace not found")


def _find_kernels_span(text: str) -> tuple[int, int]:
    marker = "KERNELS"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("KERNELS dict not found")
    brace_start = text.find("{", idx)
    if brace_start == -1:
        raise ValueError("KERNELS dict start not found")
    brace_end = _find_matching_brace(text, brace_start)
    return brace_start, brace_end


def _format_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def _insert_registry_entry(text: str, name: str, graph: str, description: str, tags: list[str], kernels: list[str]) -> str:
    brace_start, brace_end = _find_kernels_span(text)
    dict_text = text[brace_start : brace_end + 1]
    if re.search(rf'^\s*"{re.escape(name)}"\s*:', dict_text, flags=re.M):
        raise ValueError(f"kernel '{name}' already registered")

    entry = (
        f'    "{name}": {{\n'
        f'        "graph": "{graph}",\n'
        f'        "description": "{description}",\n'
        f'        "tags": {_format_list(tags)},\n'
        f'        "kernels": {_format_list(kernels)},\n'
        f"    }},\n"
    )

    insert_at = brace_end
    if not text[brace_start : brace_end + 1].strip().endswith("{"):
        # ensure entry is separated from previous if needed
        pass
    new_text = text[:insert_at] + entry + text[insert_at:]
    return new_text


def _remove_registry_entry(text: str, name: str) -> str:
    brace_start, brace_end = _find_kernels_span(text)
    dict_text = text[brace_start : brace_end + 1]
    m = re.search(rf'^\s*"{re.escape(name)}"\s*:\s*{{', dict_text, flags=re.M)
    if not m:
        raise ValueError(f"kernel '{name}' not found")
    entry_start = brace_start + m.start()
    brace_idx = brace_start + m.end() - 1
    entry_end = _find_matching_brace(text, brace_idx)
    remove_end = entry_end + 1
    while remove_end < len(text) and text[remove_end].isspace():
        if text[remove_end] == "\n":
            break
        remove_end += 1
    if remove_end < len(text) and text[remove_end] == ",":
        remove_end += 1
    # trim one trailing newline if present
    if remove_end < len(text) and text[remove_end] == "\n":
        remove_end += 1
    new_text = text[:entry_start] + text[remove_end:]
    return new_text


def cmd_make_graph(args: argparse.Namespace) -> int:
    _ensure_exp_dirs()
    name = args.name
    graph_path = KERNEL_GRAPHS_DIR / f"{name}.graph"
    backup = _backup_file(graph_path)

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
    if backup:
        print(f"backup:{backup}")
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
    existed = target.exists()
    if not existed and args.path:
        target.touch()

    backup = _backup_file(target) if existed else None
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
    if backup:
        print(f"backup:{backup}")
    print(f"{rel_target}:{start}-{end}")
    return 0


def cmd_edit_range(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    if not path.exists() and args.mode not in {"insert_before", "insert_after"}:
        raise ValueError(f"{path} does not exist")
    code = args.code
    if args.code_file:
        code = Path(args.code_file).read_text()
    if args.mode != "delete" and code is None:
        raise ValueError("code or code_file required unless mode=delete")
    backup = _backup_file(path)
    start, end = _apply_edit_range(path, args.start_line, args.end_line, code, args.mode)
    if backup:
        print(f"backup:{backup}")
    print(f"{path}:{start}-{end}")
    return 0


def cmd_delete_range(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    backup = _backup_file(path)
    start, end = _apply_edit_range(path, args.start_line, args.end_line, None, "delete")
    if backup:
        print(f"backup:{backup}")
    print(f"{path}:{start}-{end}")
    return 0


def cmd_delete_file(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    backup = _backup_file(path)
    path.unlink()
    if backup:
        print(f"backup:{backup}")
    print(f"deleted:{path}")
    return 0


def cmd_delete_graph(args: argparse.Namespace) -> int:
    graph_path = (KERNEL_GRAPHS_DIR / f"{args.name}.graph").resolve()
    backup = _backup_file(graph_path)
    if not graph_path.exists():
        raise ValueError(f"{graph_path} does not exist")
    graph_path.unlink()
    if backup:
        print(f"backup:{backup}")
    print(f"deleted:{graph_path}")
    return 0


def cmd_restore_file(args: argparse.Namespace) -> int:
    backup = Path(args.backup).resolve()
    if not backup.exists():
        raise ValueError(f"{backup} does not exist")
    target = Path(args.path).resolve()
    prev = _backup_file(target)
    shutil.copy2(backup, target)
    if prev:
        print(f"backup:{prev}")
    print(f"restored:{target}")
    return 0


def cmd_register_graph(args: argparse.Namespace) -> int:
    registry = (KERNEL_GRAPHS_DIR / "registry.py").resolve()
    text = registry.read_text()
    tags = [t for t in args.tags.split(",") if t] if args.tags else []
    kernels = [k for k in args.kernels.split(",") if k] if args.kernels else []
    if not tags:
        raise ValueError("tags must be provided (comma-separated)")
    if not kernels:
        raise ValueError("kernels must be provided (comma-separated)")
    updated = _insert_registry_entry(text, args.name, args.graph, args.description, tags, kernels)
    backup = _backup_file(registry)
    registry.write_text(updated)
    if backup:
        print(f"backup:{backup}")
    print(f"registered:{args.name}")
    return 0


def cmd_unregister_graph(args: argparse.Namespace) -> int:
    registry = (KERNEL_GRAPHS_DIR / "registry.py").resolve()
    text = registry.read_text()
    updated = _remove_registry_entry(text, args.name)
    backup = _backup_file(registry)
    registry.write_text(updated)
    if backup:
        print(f"backup:{backup}")
    print(f"unregistered:{args.name}")
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

    p_append = sub.add_parser("append-raw", help="Append raw code to experimental device/host/python")
    p_append.add_argument("--section", required=True, help="device, host, or python")
    p_append.add_argument("--code", required=True, help="Raw code to append")
    p_append.add_argument("--path", default=None, help="Explicit experimental file path to append to")
    p_append.add_argument("--start-line", type=int, default=None, help="Expected start line for append validation")
    p_append.set_defaults(func=cmd_append_raw)

    p_edit = sub.add_parser("edit-range", help="Edit a file range by line numbers")
    p_edit.add_argument("--path", required=True, help="Path to file to edit")
    p_edit.add_argument("--start-line", type=int, required=True, help="1-based start line")
    p_edit.add_argument("--end-line", type=int, required=True, help="1-based end line")
    p_edit.add_argument("--code", default=None, help="Replacement code text")
    p_edit.add_argument("--code-file", default=None, help="Read replacement code from file")
    p_edit.add_argument(
        "--mode",
        default="replace",
        choices=["replace", "delete", "insert_before", "insert_after"],
        help="Edit mode",
    )
    p_edit.set_defaults(func=cmd_edit_range)

    p_delete_range = sub.add_parser("delete-range", help="Delete a line range from a file")
    p_delete_range.add_argument("--path", required=True, help="Path to file to edit")
    p_delete_range.add_argument("--start-line", type=int, required=True, help="1-based start line")
    p_delete_range.add_argument("--end-line", type=int, required=True, help="1-based end line")
    p_delete_range.set_defaults(func=cmd_delete_range)

    p_delete_file = sub.add_parser("delete-file", help="Delete a file")
    p_delete_file.add_argument("--path", required=True, help="Path to file to delete")
    p_delete_file.set_defaults(func=cmd_delete_file)

    p_delete_graph = sub.add_parser("delete-graph", help="Delete a kernel graph file")
    p_delete_graph.add_argument("--name", required=True, help="Graph name (kernel_graphs/<name>.graph)")
    p_delete_graph.set_defaults(func=cmd_delete_graph)

    p_restore = sub.add_parser("restore-file", help="Restore a file from a backup")
    p_restore.add_argument("--backup", required=True, help="Path to .bak file")
    p_restore.add_argument("--path", required=True, help="Path to restore into")
    p_restore.set_defaults(func=cmd_restore_file)

    p_register = sub.add_parser("register-graph", help="Register a kernel graph in registry.py")
    p_register.add_argument("--name", required=True, help="Kernel name")
    p_register.add_argument("--graph", required=True, help=".graph filename")
    p_register.add_argument("--description", required=True, help="Description text")
    p_register.add_argument("--tags", required=True, help="Comma-separated tags")
    p_register.add_argument("--kernels", required=True, help="Comma-separated kernel function names")
    p_register.set_defaults(func=cmd_register_graph)

    p_unregister = sub.add_parser("unregister-graph", help="Remove a kernel from registry.py")
    p_unregister.add_argument("--name", required=True, help="Kernel name to remove")
    p_unregister.set_defaults(func=cmd_unregister_graph)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
