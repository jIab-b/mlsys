"""FlashInfer-Bench Modal Cloud Benchmark Runner."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

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
def run_benchmark(solution_json: str, config_dict: dict | None = None) -> dict:
    import traceback
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    from flashinfer_bench.compile import BuilderRegistry

    solution = Solution.model_validate_json(solution_json)
    if config_dict is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        config = BenchmarkConfig(**config_dict)

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

    # Smoke test: run first workload only; fail fast on runtime issues.
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


def _load_solution_json(solution_json_path: str) -> str:
    return Path(solution_json_path).read_text()


def _print_solution_banner(solution_json: str):
    data = json.loads(solution_json)
    name = data.get("name", "<unknown>")
    definition = data.get("definition", "<unknown>")
    print(f"Running: {name} ({definition})")


@app.local_entrypoint()
def main(solution_json_path: str = str(PROJECT_ROOT / "solution.json"), output: str = None):
    solution_json = _load_solution_json(solution_json_path)
    _print_solution_banner(solution_json)
    out = print_results(run_benchmark.remote(solution_json))
    print(out)
    if output:
        Path(output).write_text(out)
        print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--solution-json", type=str, default=str(PROJECT_ROOT / "solution.json"))
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            solution_json = _load_solution_json(args.solution_json)
            _print_solution_banner(solution_json)
            out = print_results(run_benchmark.remote(solution_json))
            print(out)
            if args.output:
                Path(args.output).write_text(out)
                print(f"Saved to {args.output}")
