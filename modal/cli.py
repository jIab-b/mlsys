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
TASKS = {"sparse_attention": "sparse_attention"}

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


def run_single(
    submission: Path,
    output: Path,
    mode: str,
    task: str,
    no_sync: bool,
    trace_dir: Path | None,
):
    task_dir = get_task_dir(task)
    tests_file = task_dir / ("benchmarks.txt" if mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")
    if not no_sync:
        sync_sglang()
    submission_code = submission.read_text()
    tests_content = tests_file.read_text()
    _log(f"Running {mode} for task '{task}' on Modal...")
    with app.run():
        result = run_eval.remote(
            submission_code,
            tests_content,
            mode,
            str(Path("/kernel_data") / "trace") if trace_dir else None,
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
    args = parser.parse_args(_strip_eval_subcommand(sys.argv[1:]))

    if not args.submission:
        parser.error("submission is required")

    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")

    output = Path(args.output)
    trace_dir = Path(args.trace) if args.trace else None
    run_single(submission, output, args.mode, args.task, args.no_sync, trace_dir)


if __name__ == "__main__":
    main()
