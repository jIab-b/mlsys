#!/usr/bin/env python3
"""CLI for running eval_suite kernels on Modal."""
import argparse
import sys
import subprocess
from pathlib import Path

from app_reuse import app, sync_sglang, run_eval, run_eval_reuse_ephemeral, run_kernel_test
from format import format_result

EVAL_ROOT = Path(__file__).parent.parent  # eval_suite/
PROJECT_ROOT = EVAL_ROOT.parent  # mlsys/

# Add paths for imports
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from references.registry import REFS

# Build TASKS from registry + legacy sparse tasks
TASKS = {
    # Legacy sparse attention tasks
    "sparse_attention": "test_kernels/sparse_attention",
    "sparse_index": "test_kernels/sparse_index",
    "sparse_attn": "test_kernels/sparse_attn",
    # DSA topk=2048 tasks
    "dsa_index_2048": "test_kernels/dsa_index_2048",
    "dsa_attn_2048": "test_kernels/dsa_attn_2048",
}
# Add all registry definitions
for name, info in REFS.items():
    TASKS[name] = f"definitions/{info['op_type']}/{name}"

CHECK = "\u2705"
CROSS = "\u274c"
STOPWATCH = "\u23f1"
ZAP = "\u26a1"
SNAIL = "\U0001F40C"
MICRO = "\u00b5"


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _format_kernel_result(result: dict, name: str, mode: str) -> None:
    """Format kernel test/bench result like eval_test."""
    system = result.get("system", {})
    gpu = system.get("gpu", "Unknown")
    returncode = result.get("returncode", -1)
    success = returncode == 0

    print(f"\"**Modal {CHECK if success else CROSS} {'success' if success else 'failure'}**")
    print(f"> {CHECK if success else CROSS} {'Benchmark' if mode == 'bench' else 'Test'} {'passed' if success else 'failed'}")
    print()
    print("Running on:")
    print(f"* GPU: `{gpu}`")
    print(f"* Torch: `{system.get('torch', 'Unknown')}`")
    print()

    if mode == "bench" and "benchmarks" in result:
        print("## Benchmarks:")
        print("```")
        for bench in result["benchmarks"]:
            print(f"{bench.get('name', 'unknown')}")
            mean = bench.get("mean_us", 0)
            std = bench.get("std_us", 0)
            best = bench.get("best_us", 0)
            worst = bench.get("worst_us", 0)
            print(f" {STOPWATCH} {mean:.1f} \u00b1 {std:.2f} {MICRO}s")
            print(f" {ZAP} {best:.1f} {MICRO}s {SNAIL} {worst:.1f} {MICRO}s")
            if bench.get("flashinfer_us"):
                speedup = bench["flashinfer_us"] / mean if mean > 0 else 0
                print(f" vs FlashInfer: {bench['flashinfer_us']:.1f} {MICRO}s ({speedup:.2f}x)")
            print()
        print("```")
    elif mode == "test" and "tests" in result:
        print("## Tests:")
        print("```")
        for test in result["tests"]:
            status = test.get("status", "unknown")
            print(f"{test.get('name', 'unknown')}")
            if status == "pass":
                print(f" {CHECK} pass")
            else:
                print(f" {CROSS} {test.get('error', 'failed')}")
            print()
        print("```")

    # Show stdout/stderr
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if stdout or stderr:
        print("## Program output:")
        print("```")
        if stdout:
            print(stdout.rstrip())
        if stderr:
            print("--- stderr ---")
            print(stderr.rstrip())
        print("```")
    print("\"")

def _strip_eval_subcommand(argv: list[str]) -> list[str]:
    if argv and argv[0] == "eval":
        return argv[1:]
    return argv


def _save_trace(result: dict, trace_dir: Path) -> None:
    archive_b64 = result.get("trace_archive_b64")
    name = result.get("trace_archive_name", "trace.tar.gz")
    if not archive_b64:
        return
    import base64
    import tarfile
    trace_dir.mkdir(parents=True, exist_ok=True)
    archive_path = trace_dir / name
    archive_path.write_bytes(base64.b64decode(archive_b64.encode("ascii")))
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(trace_dir)
    subprocess.run(["rm", str(archive_path)])

def get_task_dir(task: str) -> Path:
    return EVAL_ROOT / TASKS[task]


def generate_random_tests(task: str, count: int = 3, seed: int = 42) -> str:
    """Generate random test specs from FlashInfer workloads (axis values only)."""
    from common.workload_loader import generate_sparse_attention_specs

    # All three sparse tasks use the same input spec (batch, num_pages, seq_len)
    if task in ("sparse_attention", "sparse_index", "sparse_attn"):
        specs = generate_sparse_attention_specs(count=count, seed=seed)
    else:
        raise ValueError(f"Random tests not supported for task: {task}")

    return "\n".join(specs)


def generate_real_tests(task: str, count: int = 3, seed: int = 42) -> str:
    """Generate test specs with real workload data from safetensors."""
    from common.workload_loader import generate_real_workload_specs

    if task in ("sparse_attention", "sparse_index", "sparse_attn"):
        specs = generate_real_workload_specs(task, count=count, seed=seed)
    else:
        raise ValueError(f"Real workloads not supported for task: {task}")

    return "\n".join(specs)


def run_single(
    submission: Path,
    output: Path,
    mode: str,
    task: str,
    no_sync: bool,
    trace_dir: Path | None,
    random_tests: int | None = None,
    real_tests: int | None = None,
    random_seed: int = 42,
    ephemeral_reuse: bool = False,
):
    task_dir = get_task_dir(task)

    if real_tests is not None:
        _log(f"Generating {real_tests} REAL test(s) from FlashInfer workloads (with safetensors)...")
        tests_content = generate_real_tests(task, count=real_tests, seed=random_seed)
        _log(f"Generated {real_tests} real workload specs")
    elif random_tests is not None:
        _log(f"Generating {random_tests} random test(s) from FlashInfer workloads...")
        tests_content = generate_random_tests(task, count=random_tests, seed=random_seed)
        _log(f"Generated tests:\n{tests_content}")
    else:
        tests_file = task_dir / ("benchmarks.txt" if mode in ("benchmark", "leaderboard") else "tests.txt")
        if not tests_file.exists():
            raise FileNotFoundError(f"Tests not found: {tests_file}")
        tests_content = tests_file.read_text()

    if not no_sync:
        sync_sglang()
    submission_code = submission.read_text()
    _log(f"Running {mode} for task '{task}' on Modal...")
    solution_name = submission.stem
    with app.run():
        run_fn = run_eval_reuse_ephemeral if ephemeral_reuse else run_eval
        result = run_fn.remote(
            submission_code,
            tests_content,
            mode,
            str(Path("/kernel_data") / "trace") if trace_dir else None,
            task,  # definition_name
            solution_name,
            task,  # op_type
            task,  # task
        )
    formatted = format_result(result, mode)
    output.write_text(formatted)
    print(formatted)
    _log(f"Output saved to {output}")
    if trace_dir:
        _save_trace(result, trace_dir)


def main():
    parser = argparse.ArgumentParser(description="Run eval_suite kernels on Modal")
    parser.add_argument("submission", nargs="?", help="Submission file path")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file for formatted result")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="sparse_attention", help="Task name (use --list-tasks to see all)")
    parser.add_argument(
        "--dsa_index_2048",
        action="store_true",
        help="Shortcut for --task dsa_index_2048",
    )
    parser.add_argument(
        "--dsa_attn_2048",
        action="store_true",
        help="Shortcut for --task dsa_attn_2048",
    )
    parser.add_argument(
        "--sync",
        dest="no_sync",
        action="store_false",
        help="Sync eval_suite before running (disabled by default)",
    )
    parser.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip syncing eval_suite (default)",
    )
    parser.add_argument(
        "--ephemeral-reuse",
        action="store_true",
        help="Use ephemeral reuse eval entrypoint (no persistent cache sync)",
    )
    parser.add_argument(
        "--trace",
        nargs="?",
        const="trace",
        metavar="DIR",
        help="Save trace artifacts to local DIR (default: ./trace)",
    )
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Sample N random test inputs from FlashInfer workloads (default: 3)",
    )
    parser.add_argument(
        "--real",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Sample N REAL test inputs with safetensor data from FlashInfer (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random/--real options (default: 42)",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync eval_suite to Modal volume (no GPU required)",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Force full re-sync of eval_suite",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run kernel test file (correctness check against FlashInfer)",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Run kernel benchmark (timing against FlashInfer)",
    )
    args = parser.parse_args(_strip_eval_subcommand(sys.argv[1:]))

    # Task shortcut flags
    shortcut_tasks = []
    if args.dsa_index_2048:
        shortcut_tasks.append("dsa_index_2048")
    if args.dsa_attn_2048:
        shortcut_tasks.append("dsa_attn_2048")
    if len(shortcut_tasks) > 1:
        parser.error("Choose only one of --dsa_index_2048 or --dsa_attn_2048")
    if shortcut_tasks:
        selected_task = shortcut_tasks[0]
        default_task = parser.get_default("task")
        if args.task not in (default_task, selected_task):
            parser.error(f"--task conflicts with shortcut flag (selected '{selected_task}')")
        args.task = selected_task

    # Handle --list-tasks
    if args.list_tasks:
        print("Available tasks:")
        for name in sorted(TASKS.keys()):
            print(f"  {name}")
        return

    # Handle --test or --bench
    if args.test or args.bench:
        if not args.submission:
            parser.error("submission file is required for --test/--bench")
        kernel_file = Path(args.submission)
        if not kernel_file.exists():
            kernel_file = PROJECT_ROOT / args.submission
        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel file not found: {args.submission}")

        mode = "bench" if args.bench else "test"
        _log(f"Running kernel {mode} {kernel_file.name} on Modal...")
        kernel_code = kernel_file.read_text()
        with app.run():
            result = run_kernel_test.remote(kernel_code, kernel_file.stem, mode)

        _format_kernel_result(result, kernel_file.stem, mode)
        return

    # Handle sync-only mode (no GPU needed)
    if args.sync_only:
        from app_reuse import _sync_directory, LOCAL_EVAL_SUITE, VOLUME_EVAL_SUITE, SGLANG_MANIFEST_PATH, SYNC_EXTENSIONS
        _log("Syncing eval_suite to Modal volume (including safetensors)...")
        changed = _sync_directory(
            local_root=LOCAL_EVAL_SUITE,
            remote_root=VOLUME_EVAL_SUITE,
            manifest_path=SGLANG_MANIFEST_PATH,
            extensions=SYNC_EXTENSIONS,
            force=args.force_sync,
        )
        _log(f"Synced {changed} file(s)")
        return

    if not args.submission:
        parser.error("submission is required")

    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")

    output = Path(args.output)
    trace_dir = Path(args.trace) if args.trace else None
    if args.no_sync is None:
        args.no_sync = True

    run_single(
        submission,
        output,
        args.mode,
        args.task,
        args.no_sync,
        trace_dir,
        random_tests=args.random,
        real_tests=args.real,
        random_seed=args.seed,
        ephemeral_reuse=args.ephemeral_reuse,
    )


if __name__ == "__main__":
    main()
