#!/usr/bin/env python3
"""CLI for running eval_suite sparse attention on Modal."""
import argparse
import sys
from pathlib import Path

from app import app, sync_sglang, run_eval

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval_suite" / "sparse_attention"


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def cmd_sync(_args):
    """Sync eval_suite to Modal volume."""
    sync_sglang()


def cmd_eval(args):
    """Run eval_suite sparse attention on Modal."""
    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")

    tests_file = EVAL_ROOT / ("benchmarks.txt" if args.mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    if not args.no_sync:
        sync_sglang()

    submission_code = submission.read_text()
    tests_content = tests_file.read_text()

    _log(f"Running {args.mode} on Modal...")
    with app.run():
        result = run_eval.remote(submission_code, tests_content, args.mode)

    print(result.get("popcorn", ""))
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if stdout:
        print("--- stdout ---")
        print(stdout.rstrip("\n"))
    if stderr:
        print("--- stderr ---", file=sys.stderr)
        print(stderr.rstrip("\n"), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run eval_suite sparse attention on Modal")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_sync = subparsers.add_parser("sync", help="Sync eval_suite to Modal volume")
    p_sync.set_defaults(func=cmd_sync)

    p_eval = subparsers.add_parser("eval", help="Run eval_suite sparse attention")
    p_eval.add_argument("submission", help="Submission file path")
    p_eval.add_argument("-m", "--mode", default="test", choices=["test", "benchmark", "leaderboard", "profile"])
    p_eval.add_argument("--no-sync", action="store_true", help="Skip syncing eval_suite")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
