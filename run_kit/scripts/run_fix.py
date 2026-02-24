"""FlashInfer-Bench Modal Cloud Benchmark Runner."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

env_image = {
    "CUDA_HOME": "/usr/local/cuda",
    "TVM_FFI_CUDA_ARCH_LIST": "10.0a",
}

image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel")
    .env(env_image)
    .pip_install("flashinfer-bench", "triton", "numpy")
)

@app.function(image=image, volumes={TRACE_SET_PATH: trace_volume})
def vol_shell():
    pass


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    import traceback
    from flashinfer_bench.compile import BuilderRegistry

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{solution.definition}'")

    try:
        registry = BuilderRegistry.get_instance()
        registry.build(definition, solution)
    except Exception as e:
        return {"__error__": {"status": "COMPILE_ERROR", "log": f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"}}

    # Smoke test: run first workload only — bail fast on runtime errors
    smoke_config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    smoke_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads[:1]},
        traces={definition.name: []},
    )
    try:
        smoke_result = Benchmark(smoke_trace_set, smoke_config).run_all(dump_traces=False)
    except Exception as e:
        return {"__error__": {"status": "RUNTIME_ERROR", "log": f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"}}
    for trace in smoke_result.traces.get(definition.name, []):
        if trace.evaluation and trace.evaluation.status.value != "PASSED":
            ev = trace.evaluation
            return {"__error__": {"status": ev.status.value, "log": ev.log or repr(ev)}}

    # Full benchmark
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    try:
        result_trace_set = Benchmark(bench_trace_set, config).run_all(dump_traces=True)
    except Exception as e:
        return {"__error__": {"status": "RUNTIME_ERROR", "log": f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"}}

    results = {}
    for trace in result_trace_set.traces.get(definition.name, []):
        if not trace.evaluation:
            continue
        ev = trace.evaluation
        entry = {"status": ev.status.value, "solution": trace.solution}
        if ev.performance:
            entry.update(latency_ms=ev.performance.latency_ms, ref_latency_ms=ev.performance.reference_latency_ms, speedup=ev.performance.speedup_factor)
        if ev.correctness:
            entry.update(max_abs_err=ev.correctness.max_absolute_error, max_rel_err=ev.correctness.max_relative_error)
        if ev.log:
            entry["log"] = ev.log
        if ev.status.value != "PASSED":
            return {"__error__": {"status": ev.status.value, "log": ev.log or repr(ev)}}
        results[trace.workload.uuid] = entry
    return results


def print_results(results: dict):
    lines = []
    if "__error__" in results:
        lines.append(f"\n{results['__error__']['status']}:\n{results['__error__'].get('log', '')}")
        return "\n".join(lines)

    first_log = None
    for uid, r in results.items():
        s = r.get("status")
        line = f"  {uid[:8]}...: {s}"
        if r.get("latency_ms") is not None:
            line += f" | {r['latency_ms']:.3f}ms | {r.get('speedup', 0):.2f}x"
        if r.get("max_abs_err") is not None:
            line += f" | abs={r['max_abs_err']:.2e} rel={r.get('max_rel_err', 0):.2e}"
        lines.append(line)
        if s != "PASSED" and r.get("log"):
            if first_log is None:
                first_log = r["log"]
                lines.append(f"  LOG:\n{first_log}")
            elif r["log"] != first_log:
                lines.append(f"  LOG:\n{r['log']}")
    return "\n".join(lines) if lines else "No results."


def _load_solution(source_path: str | None = None) -> Solution:
    if source_path is None:
        from scripts.pack_solution import pack_solution

        solution_path = pack_solution()
    else:
        from scripts.pack_solution import pack_solution_from_path

        solution_path = pack_solution_from_path(Path(source_path))
    return Solution.model_validate_json(solution_path.read_text())


@app.local_entrypoint()
def main(source_path: str = None):
    solution = _load_solution(source_path)
    print(f"Running: {solution.name} ({solution.definition})")
    out = print_results(run_benchmark.remote(solution))
    print(out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Optional source file or directory path to pack into solution.json before benchmarking.",
    )
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            solution = _load_solution(args.source_path)
            print(f"Running: {solution.name} ({solution.definition})")
            out = print_results(run_benchmark.remote(solution))
            print(out)
            if args.output:
                Path(args.output).write_text(out)
                print(f"Saved to {args.output}")
