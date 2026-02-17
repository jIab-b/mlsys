"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

By default this reads `config.toml`.
You can also bypass/override config with CLI flags like `--dsa_attn`.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

try:
    import tomllib
except ImportError:
    import tomli as tomllib

app = modal.App("flashinfer-bench")
OUT_REPORT_PATH = PROJECT_ROOT / "solution" / "out.txt"

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
CUDA_HOME_PATH = "/usr/local/cuda"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

DEFINITION_ALIASES = {
    "dsa_index": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
    "dsa_attn": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gdn_decode": "gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill": "gdn_prefill_qk4_v8_d128_k_last",
    "fp8_moe": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
}


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    source_files: dict[str, bytes],
    solution_meta: dict,
    benchmark_config: dict | None = None,
    max_workloads: int = 1,
) -> dict:
    """Pack solution and run benchmark on Modal B200, returning results + packed solution."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    if benchmark_config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        config = BenchmarkConfig(**benchmark_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_root = Path(tmpdir) / "solution"
        source_root.mkdir(parents=True, exist_ok=True)

        for rel_path, content in source_files.items():
            dst = source_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)

        spec = BuildSpec(
            language=solution_meta["language"],
            target_hardware=["cuda"],
            entry_point=solution_meta["entry_point"],
        )

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

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}
    stats = {"total": len(traces), "evaluated": 0, "non_evaluated": 0}

    for i, trace in enumerate(traces):
        raw = trace.model_dump(mode="json")
        workload_obj = raw.get("workload")
        if isinstance(workload_obj, dict):
            workload_uuid = workload_obj.get("uuid") or f"trace_{i}"
        else:
            workload_uuid = f"trace_{i}"
        entry = {"solution": raw.get("solution", "")}

        evaluation = raw.get("evaluation")
        if evaluation:
            if isinstance(evaluation, dict):
                status_obj = evaluation.get("status")
                if isinstance(status_obj, dict):
                    entry["status"] = status_obj.get("value", "unknown")
                elif status_obj is not None:
                    entry["status"] = str(status_obj)
                else:
                    entry["status"] = "unknown"

                performance = evaluation.get("performance")
                if isinstance(performance, dict):
                    entry["latency_ms"] = performance.get("latency_ms")
                    entry["reference_latency_ms"] = performance.get("reference_latency_ms")
                    entry["speedup_factor"] = performance.get("speedup_factor")

                correctness = evaluation.get("correctness")
                if isinstance(correctness, dict):
                    entry["max_abs_error"] = correctness.get("max_absolute_error")
                    entry["max_rel_error"] = correctness.get("max_relative_error")
            else:
                # Some bench versions serialize evaluation directly as a status string.
                entry["status"] = str(evaluation)
                entry["detail"] = (
                    raw.get("message")
                    or raw.get("error")
                    or "Non-dict evaluation payload returned by benchmark."
                )

            stats["evaluated"] += 1
        else:
            entry["status"] = "NO_EVAL"
            entry["detail"] = (
                raw.get("status")
                or raw.get("message")
                or raw.get("error")
                or "Trace has no evaluation object (likely build/runtime failure)."
            )
            stats["non_evaluated"] += 1

        results[definition.name][workload_uuid] = entry

    return {
        "results": results,
        "stats": stats,
        "solution": {
            "name": solution.name,
            "definition": solution.definition,
            "author": solution.author,
            "language": solution_meta["language"],
        },
        "solution_json": solution.model_dump_json(indent=2),
    }


def print_results(results: dict, out_lines: list[str] | None = None):
    """Print benchmark results in a formatted way."""
    def emit(line: str):
        print(line)
        if out_lines is not None:
            out_lines.append(line)

    for def_name, traces in results.items():
        emit("")
        emit(f"{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            line = f"  Workload {workload_uuid[:8]}...: {status}"

            if result.get("latency_ms") is not None:
                line += f" | {result['latency_ms']:.3f} ms"

            if result.get("speedup_factor") is not None:
                line += f" | {result['speedup_factor']:.2f}x speedup"

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                line += f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"

            if result.get("detail"):
                line += f" | detail={result['detail']}"

            emit(line)


def _load_config_best_effort() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Warning: could not parse {config_path}: {e}")
        return {}


def _resolve_definition_alias(
    definition: str,
    dsa_index: bool,
    dsa_attn: bool,
    gdn_decode: bool,
    gdn_prefill: bool,
    fp8_moe: bool,
) -> str:
    selected = []
    if dsa_index:
        selected.append(DEFINITION_ALIASES["dsa_index"])
    if dsa_attn:
        selected.append(DEFINITION_ALIASES["dsa_attn"])
    if gdn_decode:
        selected.append(DEFINITION_ALIASES["gdn_decode"])
    if gdn_prefill:
        selected.append(DEFINITION_ALIASES["gdn_prefill"])
    if fp8_moe:
        selected.append(DEFINITION_ALIASES["fp8_moe"])

    if len(selected) > 1:
        raise ValueError("Select only one of --dsa_index/--dsa_attn/--gdn_decode/--gdn_prefill/--fp8_moe")

    if definition and selected:
        raise ValueError("Use either --definition or a shortcut flag, not both")

    if definition:
        return definition
    if selected:
        return selected[0]
    return ""


def _resolve_solution_meta(
    definition: str,
    name: str,
    author: str,
    language: str,
    entry_point: str,
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

    return {
        "definition": final_definition,
        "name": final_name,
        "author": final_author,
        "language": final_language,
        "entry_point": final_entry_point,
    }, source_dir


def _collect_source_files(source_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(source_dir).parts
        if "__pycache__" in rel_parts:
            continue

        rel_path = Path(*rel_parts).as_posix()
        files[rel_path] = path.read_bytes()

    if not files:
        raise FileNotFoundError(f"No source files found under {source_dir}")

    return files


def _normalize_entry_point(raw_entry_point: str, language: str, source_dir: Path) -> str:
    entry_point = raw_entry_point.strip()
    if not entry_point:
        raise ValueError("Entry point cannot be empty")

    if "::" in entry_point:
        return entry_point

    # Backward compatibility for old config style: entry_point = "kernel".
    if language == "cuda":
        candidates = ("binding.py", "kernel.cu")
    else:
        candidates = ("kernel.py",)

    for filename in candidates:
        if (source_dir / filename).exists():
            return f"{filename}::{entry_point}"

    return f"{candidates[0]}::{entry_point}"


def _apply_file_override(
    files: dict[str, bytes],
    source_dir: Path,
    language: str,
    file_base: str,
) -> dict[str, bytes]:
    if not file_base:
        return files

    if language != "cuda":
        raise ValueError("--file override is only supported for CUDA solutions")

    base = Path(file_base)
    if base.suffix in (".py", ".cu"):
        base = base.with_suffix("")

    py_rel = base.with_suffix(".py").as_posix()
    cu_rel = base.with_suffix(".cu").as_posix()

    if py_rel not in files:
        raise FileNotFoundError(
            f"--file override expected '{py_rel}' under {source_dir}, but it was not found"
        )
    if cu_rel not in files:
        raise FileNotFoundError(
            f"--file override expected '{cu_rel}' under {source_dir}, but it was not found"
        )

    remapped = dict(files)
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
    file_base: str = "",
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

    use_overrides = any(
        [
            selected_definition,
            name,
            author,
            language,
            entry_point,
            file_base,
        ]
    )

    try:
        if use_overrides:
            log("Packing solution from source files (override mode)...")
        else:
            log("Packing solution from source files...")

        solution_meta, source_dir = _resolve_solution_meta(
            definition=selected_definition,
            name=name,
            author=author,
            language=language,
            entry_point=entry_point,
        )

        source_files = _collect_source_files(source_dir)
        source_files = _apply_file_override(
            files=source_files,
            source_dir=source_dir,
            language=solution_meta["language"],
            file_base=file_base,
        )

        log(f"Loaded: {solution_meta['name']} ({solution_meta['definition']})")
        log(f"Entry point: {solution_meta['entry_point']}")
        if file_base:
            log(
                f"Override mode: using '{file_base}.py' as binding.py and '{file_base}.cu' as kernel.cu"
            )

        log("")
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
        report_lines.append("")
        report_lines.append("ERROR:")
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
    file: str = "",
    max_workloads: int = 1,
):
    """Pack solution and run benchmark on Modal."""
    _run(
        definition=definition,
        dsa_index=dsa_index,
        dsa_attn=dsa_attn,
        gdn_decode=gdn_decode,
        gdn_prefill=gdn_prefill,
        fp8_moe=fp8_moe,
        name=name,
        author=author,
        language=language,
        entry_point=entry_point,
        file_base=file,
        max_workloads=max_workloads,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlashInfer benchmark on Modal")
    parser.add_argument("--definition", type=str, default="", help="Override definition name")
    parser.add_argument("--dsa_index", action="store_true", help=DEFINITION_ALIASES["dsa_index"])
    parser.add_argument("--dsa_attn", action="store_true", help=DEFINITION_ALIASES["dsa_attn"])
    parser.add_argument("--gdn_decode", action="store_true", help=DEFINITION_ALIASES["gdn_decode"])
    parser.add_argument("--gdn_prefill", action="store_true", help=DEFINITION_ALIASES["gdn_prefill"])
    parser.add_argument("--fp8_moe", action="store_true", help=DEFINITION_ALIASES["fp8_moe"])
    parser.add_argument("--name", type=str, default="", help="Override solution name")
    parser.add_argument("--author", type=str, default="", help="Override solution author")
    parser.add_argument("--language", type=str, default="", help="Override build language: triton|cuda")
    parser.add_argument("--entry_point", type=str, default="", help="Override entry point")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="CUDA override base name (maps <base>.py->binding.py and <base>.cu->kernel.cu)",
    )
    parser.add_argument(
        "--max_workloads",
        type=int,
        default=1,
        help="Limit number of workloads to run (0 means all)",
    )
    args = parser.parse_args()

    with app.run():
        _run(
            definition=args.definition,
            dsa_index=args.dsa_index,
            dsa_attn=args.dsa_attn,
            gdn_decode=args.gdn_decode,
            gdn_prefill=args.gdn_prefill,
            fp8_moe=args.fp8_moe,
            name=args.name,
            author=args.author,
            language=args.language,
            entry_point=args.entry_point,
            file_base=args.file,
            max_workloads=args.max_workloads,
        )
