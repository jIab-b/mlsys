"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

By default this reads `config.toml` via `scripts.pack_solution`.
You can also bypass/override config with CLI flags like `--dsa_attn`.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet
from flashinfer_bench.agents import pack_solution_from_files

try:
    import tomllib
except ImportError:
    import tomli as tomllib

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

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
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

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

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


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


def _pack_solution_with_overrides(
    definition: str,
    name: str,
    author: str,
    language: str,
    entry_point: str,
) -> Solution:
    config = _load_config_best_effort()
    solution_cfg = config.get("solution", {})
    build_cfg = config.get("build", {})

    final_definition = definition or solution_cfg.get("definition", "")
    if not final_definition:
        raise ValueError("No definition provided. Use a shortcut flag, --definition, or a valid config.toml")

    final_language = language or build_cfg.get("language", "triton")
    if final_language not in ("triton", "cuda"):
        raise ValueError(f"Unsupported language: {final_language}")

    final_entry_point = entry_point or build_cfg.get("entry_point", "kernel")
    final_name = name or solution_cfg.get("name", f"{final_definition}-dev")
    final_author = author or solution_cfg.get("author", "unknown")

    source_dir = PROJECT_ROOT / "solution" / ("triton" if final_language == "triton" else "cuda")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    spec = BuildSpec(
        language=final_language,
        target_hardware=["cuda"],
        entry_point=final_entry_point,
    )

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=final_name,
        definition=final_definition,
        author=final_author,
    )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {solution_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {final_language}")

    return solution


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
):
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
        ]
    )

    if use_overrides:
        print("Packing solution from source files (override mode)...")
        solution = _pack_solution_with_overrides(
            definition=selected_definition,
            name=name,
            author=author,
            language=language,
            entry_point=entry_point,
        )
    else:
        from scripts.pack_solution import pack_solution

        print("Packing solution from source files...")
        solution_path = pack_solution()
        print("\nLoading solution...")
        solution = Solution.model_validate_json(solution_path.read_text())

    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


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
        )
