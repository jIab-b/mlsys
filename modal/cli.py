#!/usr/bin/env python3
"""CLI for running eval_suite sparse attention on Modal."""
import argparse
import sys
from pathlib import Path

from app import app, sync_sglang, run_eval
from format import format_result

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval_suite"
TASKS = {"sparse_attention": "sparse_attention"}

def _log(message: str) -> None:
    print(message, file=sys.stderr)


def get_task_dir(task: str) -> Path:
    return EVAL_ROOT / TASKS[task]


def run_single(submission: Path, output: Path, mode: str, task: str, no_sync: bool):
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
        result = run_eval.remote(submission_code, tests_content, mode)
    formatted = format_result(result, mode)
    output.write_text(formatted)
    print(formatted)
    _log(f"Output saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Run eval_suite kernels on Modal")
    parser.add_argument("submission", nargs="?", help="Submission file path")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file for formatted result")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="sparse_attention", choices=list(TASKS.keys()))
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing eval_suite")
    args = parser.parse_args()

    if not args.submission:
        parser.error("submission is required")

    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")

    output = Path(args.output)
    run_single(submission, output, args.mode, args.task, args.no_sync)


if __name__ == "__main__":
    main()
