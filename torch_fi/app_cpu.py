#!/usr/bin/env python3
"""Single-file Modal runner: CPU compile, GPU benchmark (FlashInfer-Bench), source passed as .py path.

Design goals:
- No persisted solution.json on disk.
- Compile torch extension artifacts on CPU first.
- Reuse compiled cache on GPU to minimize GPU active time.
- Print full COMPILE_ERROR / RUNTIME_ERROR logs to local stdout.

Example:
    modal run torch_fi/app_cpu.py -- \
      --sol tests/dsa_attn2.py \
      --definition dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
      --entry-point submission.py::custom_kernel \
      --author local
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import modal


APP_NAME = "flashinfer-bench"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
GPU_TYPE = "B200"

TRACE_VOL_NAME = "flashinfer-trace"
TRACE_PATH = "/data"
CACHE_ROOT = f"{TRACE_PATH}/.torch_fi_cache"

LOCAL_EXT_CACHE = "/tmp/torch_ext"


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TVM_FFI_CUDA_ARCH_LIST": "10.0a",
            "FIB_DATASET_PATH": TRACE_PATH,
        }
    )
    .pip_install("flashinfer-bench", "triton", "numpy")
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && "
        "echo /usr/local/cuda/lib64/stubs > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig"
    )
)

app = modal.App(APP_NAME)
trace_volume = modal.Volume.from_name(TRACE_VOL_NAME, create_if_missing=False)
VOLUMES = {TRACE_PATH: trace_volume}


def _sha_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _normalize_definition(def_value: str) -> str:
    """Normalize short aliases to real FlashInfer definition ids."""
    key = def_value.strip()
    alias_map = {
        "dsa_index": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        "dsa_sparse_index_2048": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        "dsa_attn": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
        "dsa_sparse_attention_2048": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    }
    return alias_map.get(key, key)


def _safe_rm(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def _copytree_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    _safe_rm(dst)
    shutil.copytree(src, dst)
    return True


def _mk_solution_from_source(
    submission_code: str,
    definition: str,
    entry_point: str,
    language: str,
    binding: str | None,
    name: str,
    author: str,
):
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    with tempfile.TemporaryDirectory(prefix="torch_fi_src_") as td:
        src_root = Path(td)
        (src_root / "submission.py").write_text(submission_code)

        sig = inspect.signature(BuildSpec)
        kwargs: dict[str, Any] = {
            "language": language,
            "target_hardware": ["cuda"],
            "entry_point": entry_point,
        }
        if "binding" in sig.parameters and binding is not None:
            kwargs["binding"] = binding

        spec = BuildSpec(**kwargs)

        solution = pack_solution_from_files(
            path=str(src_root),
            spec=spec,
            name=name,
            definition=definition,
            author=author,
        )
    return solution


def _run_compile_kernel(submission_code: str, torch_ext_dir: str) -> dict:
    """Run compile_kernel() from submission.py to prebuild torch extensions."""
    with tempfile.TemporaryDirectory(prefix="torch_fi_compile_") as td:
        submission_path = Path(td) / "submission.py"
        submission_path.write_text(submission_code)
        env = os.environ.copy()
        env["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, pathlib; "
                    "p=pathlib.Path('submission.py').resolve(); "
                    "s=importlib.util.spec_from_file_location('submission', p); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                    "fn=getattr(m, 'compile_kernel', None); "
                    "print('compile_kernel:found' if fn else 'compile_kernel:missing'); "
                    "fn() if fn else None"
                ),
            ],
            cwd=td,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "compile_kernel failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return {"stdout": proc.stdout, "stderr": proc.stderr}


def _make_adapter_source(submission_code: str, input_names: list[str], num_outputs: int) -> str:
    args = ", ".join(input_names)
    tuple_expr = ", ".join(input_names)
    if len(input_names) == 1:
        tuple_expr += ","
    if num_outputs == 1:
        ret_logic = (
            "    if isinstance(out, tuple):\n"
            "        return out[0]\n"
            "    return out\n"
        )
    else:
        ret_logic = (
            "    if isinstance(out, tuple):\n"
            "        return out\n"
            "    raise RuntimeError('custom_kernel must return a tuple for multi-output definitions')\n"
        )
    return (
        submission_code
        + "\n\n"
        + f"def fib_entry({args}):\n"
        + f"    data = ({tuple_expr})\n"
        + "    out = custom_kernel(data)\n"
        + ret_logic
    )


def _mk_python_solution_from_source(
    submission_code: str,
    definition: str,
    input_names: list[str],
    num_outputs: int,
    name: str,
    author: str,
):
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    with tempfile.TemporaryDirectory(prefix="torch_fi_src_") as td:
        src_root = Path(td)
        adapter_source = _make_adapter_source(submission_code, input_names, num_outputs)
        (src_root / "submission.py").write_text(adapter_source)

        sig = inspect.signature(BuildSpec)
        kwargs: dict[str, Any] = {
            "language": "python",
            "target_hardware": ["cuda"],
            "entry_point": "submission.py::fib_entry",
        }
        if "destination_passing_style" in sig.parameters:
            kwargs["destination_passing_style"] = False

        spec = BuildSpec(**kwargs)

        solution = pack_solution_from_files(
            path=str(src_root),
            spec=spec,
            name=name,
            definition=definition,
            author=author,
        )
    return solution


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def compile_cpu(
    submission_code: str,
    cache_key: str,
    definition: str,
    entry_point: str,
    language: str,
    binding: str | None,
    solution_name: str,
    author: str,
) -> dict:
    """Compile on CPU-only container and persist torch extension cache under /data."""
    try:
        from flashinfer_bench import TraceSet
        remote_cache_dir = f"{CACHE_ROOT}/{cache_key}"
        _safe_rm(LOCAL_EXT_CACHE)
        _safe_rm(remote_cache_dir)

        trace_set = TraceSet.from_path(TRACE_PATH)
        if definition not in trace_set.definitions:
            raise ValueError(f"Definition '{definition}' not found in dataset at {TRACE_PATH}")

        compile_log = _run_compile_kernel(submission_code, LOCAL_EXT_CACHE)

        copied = _copytree_if_exists(LOCAL_EXT_CACHE, remote_cache_dir)
        trace_volume.commit()

        return {
            "ok": True,
            "status": "OK",
            "cache_key": cache_key,
            "cache_saved": copied,
            "cache_dir": remote_cache_dir,
            "compile_kernel": compile_log,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "COMPILE_ERROR",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


@app.function(image=image, volumes=VOLUMES, gpu=f"{GPU_TYPE}:1", timeout=3600)
def run_gpu(
    submission_code: str,
    cache_key: str,
    definition: str,
    entry_point: str,
    language: str,
    binding: str | None,
    solution_name: str,
    author: str,
    workload_limit: int,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
) -> dict:
    """Run FlashInfer-Bench on GPU, restoring CPU-built cache if available."""
    try:
        cached_dir = f"{CACHE_ROOT}/{cache_key}"
        restored = _copytree_if_exists(cached_dir, LOCAL_EXT_CACHE)
        if not restored:
            return {
                "ok": False,
                "status": "RUNTIME_ERROR",
                "error": f"Cache miss at {cached_dir}; run CPU compile first.",
            }

        os.environ["TORCH_EXTENSIONS_DIR"] = LOCAL_EXT_CACHE
        from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

        trace_set = TraceSet.from_path(TRACE_PATH)
        if definition not in trace_set.definitions:
            raise ValueError(f"Definition '{definition}' not found in dataset at {TRACE_PATH}")

        defn = trace_set.definitions[definition]
        input_names = list(defn.inputs.keys())
        num_outputs = len(defn.outputs)
        solution = _mk_python_solution_from_source(
            submission_code=submission_code,
            definition=definition,
            input_names=input_names,
            num_outputs=num_outputs,
            name=solution_name,
            author=author,
        )
        workloads = list(trace_set.workloads.get(definition, []))
        if not workloads:
            raise ValueError(f"No workloads found for definition '{definition}'")

        if workload_limit > 0:
            workloads = workloads[:workload_limit]

        bench_ts = TraceSet(
            root=trace_set.root,
            definitions={defn.name: defn},
            solutions={defn.name: [solution]},
            workloads={defn.name: workloads},
            traces={defn.name: []},
        )

        cfg = BenchmarkConfig(warmup_runs=warmup_runs, iterations=iterations, num_trials=num_trials)
        out_ts = Benchmark(bench_ts, cfg).run_all(dump_traces=True)

        results = {}
        first_error = None
        for tr in out_ts.traces.get(defn.name, []):
            ev = tr.evaluation
            if ev is None:
                continue
            entry = {
                "status": ev.status.value,
                "workload": tr.workload.uuid,
            }
            if ev.log:
                entry["log"] = ev.log
            if ev.performance:
                entry["latency_ms"] = ev.performance.latency_ms
                entry["speedup"] = ev.performance.speedup_factor
                entry["ref_latency_ms"] = ev.performance.reference_latency_ms
            if ev.correctness:
                entry["max_abs_err"] = ev.correctness.max_absolute_error
                entry["max_rel_err"] = ev.correctness.max_relative_error

            results[tr.workload.uuid] = entry
            if ev.status.value != "PASSED" and first_error is None:
                first_error = {
                    "status": ev.status.value,
                    "log": ev.log or repr(ev),
                }

        if first_error is not None:
            return {
                "ok": False,
                "status": "RUNTIME_ERROR",
                "error": first_error["status"],
                "log": first_error["log"],
                "cache_restored": restored,
                "results": results,
            }

        return {
            "ok": True,
            "status": "OK",
            "cache_restored": restored,
            "results": results,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "RUNTIME_ERROR",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _print_json_block(title: str, obj: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, default=str))


def _print_failure(prefix: str, payload: dict) -> None:
    # Intentionally print to stdout (not stderr), as requested.
    print(f"\n{prefix}: {payload.get('status', 'ERROR')}")
    if payload.get("error"):
        print(payload["error"])
    if payload.get("log"):
        print(payload["log"])
    if payload.get("traceback"):
        print(payload["traceback"])
    if payload.get("results"):
        _print_json_block("Partial Results", payload["results"])


@app.local_entrypoint()
def main(
    sol: str,
    definition: str = "dsa_sparse_index_2048",
    def_: str = "",
    entry_point: str = "submission.py::custom_kernel",
    language: str = "cuda",
    binding: str = "torch",
    name: str = "torch-fi-inline",
    author: str = "local",
    workload_limit: int = 5,
    warmup_runs: int = 1,
    iterations: int = 10,
    num_trials: int = 1,
    output: str = "",
    o: str = "",
    compile_only: bool = False,
):
    src_path = Path(sol).expanduser().resolve()
    if not src_path.exists():
        print(f"ERROR: solution path not found: {src_path}")
        raise SystemExit(2)

    requested_def = def_ if def_ else definition
    normalized_definition = _normalize_definition(requested_def)
    output_path = o if o else output

    submission_code = src_path.read_text()
    cache_key = _sha_key(submission_code, normalized_definition, entry_point, language, str(binding))

    print("Running CPU compile stage...")
    c = compile_cpu.remote(
        submission_code=submission_code,
        cache_key=cache_key,
        definition=normalized_definition,
        entry_point=entry_point,
        language=language,
        binding=binding,
        solution_name=name,
        author=author,
    )

    _print_json_block("Compile Result", c)
    if not c.get("ok"):
        _print_failure("COMPILE_ERROR", c)
        if output_path:
            Path(output_path).write_text(json.dumps({"compile": c}, indent=2, default=str))
        raise SystemExit(1)

    if compile_only:
        print("Compile-only mode complete.")
        if output_path:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "sol": str(src_path),
                        "definition": normalized_definition,
                        "compile": c,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"Saved to {output_path}")
        return

    print("\nRunning GPU benchmark stage...")
    r = run_gpu.remote(
        submission_code=submission_code,
        cache_key=cache_key,
        definition=normalized_definition,
        entry_point=entry_point,
        language=language,
        binding=binding,
        solution_name=name,
        author=author,
        workload_limit=workload_limit,
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )

    _print_json_block("Run Result", r)
    if not r.get("ok"):
        _print_failure("RUNTIME_ERROR", r)
        if output_path:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "sol": str(src_path),
                        "definition": normalized_definition,
                        "compile": c,
                        "run": r,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"Saved to {output_path}")
        raise SystemExit(1)

    # Friendly compact summary
    results = r.get("results", {})
    if not results:
        print("No results returned.")
        return

    print("\nWorkloads:")
    for uid, entry in results.items():
        line = f"  {uid[:8]}...: {entry.get('status')}"
        if entry.get("latency_ms") is not None:
            line += f" | {entry['latency_ms']:.3f} ms"
        if entry.get("speedup") is not None:
            line += f" | {entry['speedup']:.2f}x"
        print(line)

    if output_path:
        Path(output_path).write_text(
            json.dumps(
                {
                    "sol": str(src_path),
                    "definition": normalized_definition,
                    "compile": c,
                    "run": r,
                },
                indent=2,
                default=str,
            )
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sol", "--sol", required=True)
    parser.add_argument("-def", "--def", "--definition", dest="def_", default="dsa_sparse_index_2048")
    parser.add_argument("-entry", "--entry-point", default="submission.py::custom_kernel")
    parser.add_argument("-lang", "--language", default="cuda")
    parser.add_argument("-bind", "--binding", default="torch")
    parser.add_argument("-name", "--name", default="torch-fi-inline")
    parser.add_argument("-author", "--author", default="local")
    parser.add_argument("-wl", "--workload-limit", type=int, default=5)
    parser.add_argument("-warm", "--warmup-runs", type=int, default=1)
    parser.add_argument("-it", "--iterations", type=int, default=10)
    parser.add_argument("-trials", "--num-trials", type=int, default=1)
    parser.add_argument("-o", "--o", "--output", dest="o", default="")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            main(
                sol=args.sol,
                def_=args.def_,
                entry_point=args.entry_point,
                language=args.language,
                binding=args.binding,
                name=args.name,
                author=args.author,
                workload_limit=args.workload_limit,
                warmup_runs=args.warmup_runs,
                iterations=args.iterations,
                num_trials=args.num_trials,
                o=args.o,
                compile_only=args.compile_only,
            )
