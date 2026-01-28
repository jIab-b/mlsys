#!/usr/bin/env python3
"""CLI for running eval_suite sparse attention on Modal."""
import argparse
import sys
import subprocess
from pathlib import Path

from app import app, sync_sglang, run_eval
from format import format_result

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval_suite"
TASKS = {
    "sparse_attention": "test_kernels/sparse_attention",
    "sparse_index": "test_kernels/sparse_index",
    "sparse_attn": "test_kernels/sparse_attn",
}

# Add eval_suite to path for workload_loader
sys.path.insert(0, str(EVAL_ROOT))

def _log(message: str) -> None:
    print(message, file=sys.stderr)

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
    """Generate random test specs from FlashInfer workloads."""
    from common.workload_loader import generate_sparse_attention_specs

    # All three tasks use the same input spec (batch, num_pages, seq_len)
    if task in ("sparse_attention", "sparse_index", "sparse_attn"):
        specs = generate_sparse_attention_specs(count=count, seed=seed)
    else:
        raise ValueError(f"Random tests not supported for task: {task}")

    return "\n".join(specs)


def run_single(
    submission: Path,
    output: Path,
    mode: str,
    task: str,
    no_sync: bool,
    trace_dir: Path | None,
    random_tests: int | None = None,
    random_seed: int = 42,
):
    task_dir = get_task_dir(task)

    if random_tests is not None:
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
        result = run_eval.remote(
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
    parser.add_argument("-t", "--task", default="sparse_attention", choices=list(TASKS.keys()))
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing eval_suite")
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random option (default: 42)",
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
    args = parser.parse_args(_strip_eval_subcommand(sys.argv[1:]))

    # Handle sync-only mode (no GPU needed)
    if args.sync_only:
        from app import _sync_directory, LOCAL_EVAL_SUITE, VOLUME_EVAL_SUITE, SGLANG_MANIFEST_PATH, SYNC_EXTENSIONS
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
    run_single(
        submission,
        output,
        args.mode,
        args.task,
        args.no_sync,
        trace_dir,
        random_tests=args.random,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
