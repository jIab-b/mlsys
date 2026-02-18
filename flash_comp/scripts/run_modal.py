import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import modal

try:
    import tomllib
except ImportError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("flashinfer-bench")
OUT_REPORT_PATH = PROJECT_ROOT / "solution" / "out.txt"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
UV_PACKAGES = [
    "flashinfer-bench",
    "numpy",
    "nvidia-cutlass-dsl",
    "nvtx",
    "pydot",
]

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .run_commands("pip install --upgrade pip uv")
    .run_commands("uv pip install --system " + " ".join(UV_PACKAGES))
)

DEFINITION_ALIASES = {
    "dsa_index": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
    "dsa_attn": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gdn_decode": "gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill": "gdn_prefill_qk4_v8_d128_k_last",
    "fp8_moe": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
}

WORKER_LOG_DIRS = (
    Path("/root/.cache/flashinfer_bench/cache/logs"),
    Path("/tmp/flashinfer_bench"),
)


@app.function(image=image, gpu="B200:1", timeout=90, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    source_files: dict[str, bytes],
    solution_meta: dict,
    benchmark_config: dict | None = None,
    max_workloads: int = 1,
) -> dict:
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    if benchmark_config is not None:
        config = BenchmarkConfig(**benchmark_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_root = Path(tmpdir) / "solution"
        source_root.mkdir(parents=True, exist_ok=True)

        for rel_path, content in source_files.items():
            dst = source_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)

        spec_kwargs: dict[str, object] = {
            "language": solution_meta["language"],
            "target_hardware": ["cuda"],
            "entry_point": solution_meta["entry_point"],
        }
        binding = solution_meta.get("binding", "")

        # Torch + Python entrypoint (binding.py::kernel) should use PythonBuilder.
        entry_file = str(solution_meta["entry_point"]).split("::", 1)[0]
        if binding == "torch" and entry_file.endswith(".py"):
            spec_kwargs["language"] = "python"
        elif binding:
            spec_kwargs["binding"] = binding

        try:
            spec = BuildSpec(**spec_kwargs)
        except TypeError:
            # Older flashinfer_bench versions may not expose BuildSpec.binding.
            spec_kwargs.pop("binding", None)
            spec = BuildSpec(**spec_kwargs)

        solution = pack_solution_from_files(
            path=str(source_root),
            spec=spec,
            name=solution_meta["name"],
            definition=solution_meta["definition"],
            author=solution_meta["author"],
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")
    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    result_trace_set = Benchmark(bench_trace_set, config).run_all(dump_traces=True)
    traces = result_trace_set.traces.get(definition.name, [])

    results = {definition.name: {}}
    stats = {"total": len(traces), "evaluated": 0, "non_evaluated": 0}

    for i, trace in enumerate(traces):
        raw = trace.model_dump(mode="json")
        workload = raw.get("workload")
        workload_uuid = workload.get("uuid") if isinstance(workload, dict) else None
        workload_uuid = workload_uuid or f"trace_{i}"

        entry = {"solution": raw.get("solution", "")}
        evaluation = raw.get("evaluation")

        if isinstance(evaluation, dict):
            status = evaluation.get("status")
            if isinstance(status, dict):
                entry["status"] = status.get("value", "unknown")
            elif status is None:
                entry["status"] = "unknown"
            else:
                entry["status"] = str(status)

            performance = evaluation.get("performance")
            if isinstance(performance, dict):
                entry["latency_ms"] = performance.get("latency_ms")
                entry["reference_latency_ms"] = performance.get("reference_latency_ms")
                entry["speedup_factor"] = performance.get("speedup_factor")

            correctness = evaluation.get("correctness")
            if isinstance(correctness, dict):
                entry["max_abs_error"] = correctness.get("max_absolute_error")
                entry["max_rel_error"] = correctness.get("max_relative_error")

            for key in ("message", "error", "log"):
                value = evaluation.get(key)
                if value:
                    entry[f"evaluation_{key}"] = str(value)

            stats["evaluated"] += 1
        elif evaluation is not None:
            entry["status"] = str(evaluation)
            entry["detail"] = "Non-dict evaluation payload returned by benchmark."
            stats["evaluated"] += 1
        else:
            entry["status"] = "NO_EVAL"
            entry["detail"] = raw.get("status") or "Trace has no evaluation object (likely build/runtime failure)."
            stats["non_evaluated"] += 1

        _attach_runtime_details(entry)
        results[definition.name][workload_uuid] = entry

    return {
        "results": results,
        "stats": stats,
        "solution": {
            "name": solution.name,
            "definition": solution.definition,
            "author": solution.author,
            "language": solution_meta["language"],
            "binding": solution_meta.get("binding") or "",
        },
        "solution_json": solution.model_dump_json(indent=2),
    }


def _read_latest_worker_log(solution_name: str) -> tuple[str | None, str | None]:
    if not solution_name:
        return None, None

    candidates: list[Path] = []
    for log_dir in WORKER_LOG_DIRS:
        if log_dir.exists():
            candidates.extend(log_dir.glob(f"{solution_name}_*.log"))

    if not candidates:
        return None, None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return str(latest), latest.read_text(errors="replace")
    except Exception:
        return str(latest), None


def _attach_runtime_details(entry: dict) -> None:
    status = str(entry.get("status", "")).upper()
    if status != "NO_EVAL" and "ERROR" not in status:
        return

    eval_log = str(entry.get("evaluation_log", "")).strip()
    if eval_log:
        entry["error_detail"] = eval_log
        return

    log_path, log_text = _read_latest_worker_log(entry.get("solution", ""))
    if log_path:
        entry["worker_log_path"] = log_path
    if log_text:
        entry["error_detail"] = log_text.rstrip()


def print_results(results: dict, out_lines: list[str] | None = None):
    def emit(line: str = ""):
        print(line)
        if out_lines is not None:
            out_lines.append(line)

    for def_name, traces in results.items():
        emit()
        emit(f"{def_name}:")
        for workload_uuid, result in traces.items():
            line = f"  Workload {workload_uuid[:8]}...: {result.get('status')}"
            if result.get("latency_ms") is not None:
                line += f" | {result['latency_ms']:.3f} ms"
            if result.get("speedup_factor") is not None:
                line += f" | {result['speedup_factor']:.2f}x speedup"
            if result.get("max_abs_error") is not None:
                rel_err = result.get("max_rel_error", 0)
                line += f" | abs_err={result['max_abs_error']:.2e}, rel_err={rel_err:.2e}"
            emit(line)

            for key in ("detail", "evaluation_message", "evaluation_error"):
                value = result.get(key)
                if value:
                    emit(f"    {key}: {value}")

            if result.get("error_detail"):
                emit("    error_detail:")
                for subline in str(result["error_detail"]).splitlines():
                    emit(f"      {subline}")

            if result.get("worker_log_path"):
                emit(f"    worker_log_path: {result['worker_log_path']}")


def _load_config_best_effort() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        print(f"Warning: could not parse {config_path}: {exc}")
        return {}


def _resolve_definition_alias(
    definition: str,
    dsa_index: bool,
    dsa_attn: bool,
    gdn_decode: bool,
    gdn_prefill: bool,
    fp8_moe: bool,
) -> str:
    flags = {
        "dsa_index": dsa_index,
        "dsa_attn": dsa_attn,
        "gdn_decode": gdn_decode,
        "gdn_prefill": gdn_prefill,
        "fp8_moe": fp8_moe,
    }
    selected = [DEFINITION_ALIASES[flag] for flag, enabled in flags.items() if enabled]

    if len(selected) > 1:
        raise ValueError("Select only one of --dsa_index/--dsa_attn/--gdn_decode/--gdn_prefill/--fp8_moe")
    if definition and selected:
        raise ValueError("Use either --definition or a shortcut flag, not both")
    if definition:
        return definition
    if selected:
        return selected[0]
    return ""


def _normalize_binding(raw_binding: str) -> str:
    binding = (raw_binding or "").strip().lower()
    if not binding:
        return ""
    if binding in ("tvm-ffi", "tvm_ffi"):
        return "tvm-ffi"
    if binding == "torch":
        return "torch"
    raise ValueError(f"Unsupported binding: {raw_binding}. Use 'tvm-ffi' or 'torch'.")


def _normalize_entry_point(raw_entry_point: str, language: str, source_dir: Path) -> str:
    entry_point = raw_entry_point.strip()
    if not entry_point:
        raise ValueError("Entry point cannot be empty")
    if "::" in entry_point:
        return entry_point

    candidates = ("binding.py", "kernel.cu") if language == "cuda" else ("kernel.py",)
    for filename in candidates:
        if (source_dir / filename).exists():
            return f"{filename}::{entry_point}"
    return f"{candidates[0]}::{entry_point}"


def _resolve_solution_meta(
    definition: str,
    name: str,
    author: str,
    language: str,
    entry_point: str,
    binding: str,
) -> tuple[dict, Path]:
    config = _load_config_best_effort()
    solution_cfg = config.get("solution", {})
    build_cfg = config.get("build", {})

    final_definition = definition or solution_cfg.get("definition", "")
    if not final_definition:
        raise ValueError("No definition provided. Use a shortcut flag, --definition, or a valid config.toml")

    final_language = language or build_cfg.get("language", "triton")
    if final_language not in ("triton", "cuda"):
        raise ValueError(f"Unsupported language: {final_language}")

    final_name = name or solution_cfg.get("name", f"{final_definition}-dev")
    final_author = author or solution_cfg.get("author", "unknown")
    source_dir = PROJECT_ROOT / "solution" / ("triton" if final_language == "triton" else "cuda")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    raw_entry_point = entry_point or build_cfg.get("entry_point", "kernel")
    final_entry_point = _normalize_entry_point(raw_entry_point, final_language, source_dir)

    final_binding = _normalize_binding(binding or build_cfg.get("binding", ""))
    if final_language == "cuda" and not final_binding:
        final_binding = "torch"

    return {
        "definition": final_definition,
        "name": final_name,
        "author": final_author,
        "language": final_language,
        "entry_point": final_entry_point,
        "binding": final_binding,
    }, source_dir


def _collect_source_files(source_dir: Path) -> dict[str, bytes]:
    files = {
        path.relative_to(source_dir).as_posix(): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(source_dir).parts
    }
    if not files:
        raise FileNotFoundError(f"No source files found under {source_dir}")
    return files


def _apply_file_override(
    files: dict[str, bytes],
    source_dir: Path,
    language: str,
    file: str,
) -> dict[str, bytes]:
    if not file:
        return files
    if language != "cuda":
        raise ValueError("--file override is only supported for CUDA solutions")

    base = Path(file)
    if base.suffix in (".py", ".cu"):
        base = base.with_suffix("")

    py_rel = base.with_suffix(".py").as_posix()
    cu_rel = base.with_suffix(".cu").as_posix()
    if py_rel not in files:
        raise FileNotFoundError(f"--file override expected '{py_rel}' under {source_dir}, but it was not found")
    if cu_rel not in files:
        raise FileNotFoundError(f"--file override expected '{cu_rel}' under {source_dir}, but it was not found")

    remapped = {path: content for path, content in files.items() if not path.endswith(".cu")}
    remapped["binding.py"] = files[py_rel]
    remapped["kernel.cu"] = files[cu_rel]
    return remapped


def _run(
    definition: str = "",
    dsa_index: bool = False,
    dsa_attn: bool = False,
    gdn_decode: bool = False,
    gdn_prefill: bool = False,
    fp8_moe: bool = False,
    name: str = "",
    author: str = "",
    language: str = "",
    entry_point: str = "",
    binding: str = "",
    file: str = "",
    max_workloads: int = 1,
):
    report_lines: list[str] = []

    def log(line: str = ""):
        print(line)
        report_lines.append(line)

    def write_report():
        OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT_PATH.write_text("\n".join(report_lines).rstrip() + "\n")
        print(f"Saved run report: {OUT_REPORT_PATH}")

    selected_definition = _resolve_definition_alias(
        definition=definition,
        dsa_index=dsa_index,
        dsa_attn=dsa_attn,
        gdn_decode=gdn_decode,
        gdn_prefill=gdn_prefill,
        fp8_moe=fp8_moe,
    )

    use_overrides = any((selected_definition, name, author, language, entry_point, binding, file))

    try:
        log("Packing solution from source files (override mode)..." if use_overrides else "Packing solution from source files...")

        solution_meta, source_dir = _resolve_solution_meta(
            definition=selected_definition,
            name=name,
            author=author,
            language=language,
            entry_point=entry_point,
            binding=binding,
        )

        source_files = _apply_file_override(
            files=_collect_source_files(source_dir),
            source_dir=source_dir,
            language=solution_meta["language"],
            file=file,
        )

        log(f"Loaded: {solution_meta['name']} ({solution_meta['definition']})")
        log(f"Entry point: {solution_meta['entry_point']}")
        log(f"Binding: {solution_meta.get('binding') or '(default)'}")
        if file:
            log(f"Override mode: using '{file}.py' as binding.py and '{file}.cu' as kernel.cu")

        log("Source files packed:")
        for rel_path in sorted(source_files):
            log(f"  - {rel_path}")

        log()
        log("Running benchmark on Modal B200...")
        payload = run_benchmark.remote(source_files, solution_meta, None, max_workloads)

        solution_path = PROJECT_ROOT / "solution.json"
        solution_path.write_text(payload["solution_json"])

        packed_solution = payload["solution"]
        log(f"Solution packed: {solution_path}")
        log(f"  Name: {packed_solution['name']}")
        log(f"  Definition: {packed_solution['definition']}")
        log(f"  Author: {packed_solution['author']}")
        log(f"  Language: {packed_solution['language']}")
        log(f"  Binding: {packed_solution.get('binding') or '(default)'}")

        stats = payload.get("stats", {})
        if stats:
            log(
                f"Trace stats: total={stats.get('total', 0)}, "
                f"evaluated={stats.get('evaluated', 0)}, "
                f"non_evaluated={stats.get('non_evaluated', 0)}"
            )

        results = payload.get("results", {})
        if not results or all(not traces for traces in results.values()):
            log("No evaluated trace results returned (empty evaluation set).")
            return

        print_results(results, out_lines=report_lines)
    except Exception:
        report_lines.extend(["", "ERROR:"])
        report_lines.extend(traceback.format_exc().rstrip().splitlines())
        raise
    finally:
        write_report()


@app.local_entrypoint()
def main(
    definition: str = "",
    dsa_index: bool = False,
    dsa_attn: bool = False,
    gdn_decode: bool = False,
    gdn_prefill: bool = False,
    fp8_moe: bool = False,
    name: str = "",
    author: str = "",
    language: str = "",
    entry_point: str = "",
    binding: str = "",
    file: str = "",
    max_workloads: int = 1,
):
    _run(**locals())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlashInfer benchmark on Modal")
    parser.add_argument("--definition", type=str, default="", help="Override definition name")
    for flag_name in DEFINITION_ALIASES:
        parser.add_argument(
            f"--{flag_name.replace('_', '-')}",
            f"--{flag_name}",
            dest=flag_name,
            action="store_true",
            help=DEFINITION_ALIASES[flag_name],
        )
    parser.add_argument("--name", type=str, default="", help="Override solution name")
    parser.add_argument("--author", type=str, default="", help="Override solution author")
    parser.add_argument("--language", type=str, default="", help="Override build language: triton|cuda")
    parser.add_argument("--entry-point", "--entry_point", dest="entry_point", type=str, default="", help="Override entry point")
    parser.add_argument("--binding", type=str, default="", help="Override binding: tvm-ffi|torch")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="CUDA override base name (maps <base>.py->binding.py and <base>.cu->kernel.cu)",
    )
    parser.add_argument(
        "--max-workloads",
        "--max_workloads",
        dest="max_workloads",
        type=int,
        default=1,
        help="Limit number of workloads to run (0 means all)",
    )
    return parser


if __name__ == "__main__":
    args = vars(_build_parser().parse_args())
    with app.run():
        _run(**args)
