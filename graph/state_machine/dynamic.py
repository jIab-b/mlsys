from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DynamicRunResult:
    label: str
    command: List[str]
    returncode: int
    timed_out: bool
    output_path: Path


def _modal_cli(repo_root: Path) -> Path:
    return repo_root / "tests" / "eval_test" / "modal" / "cli.py"


def run_modal_eval(
    *,
    repo_root: Path,
    task: str,
    mode: str,
    submission: Path,
    output_path: Path,
    timeout_seconds: int = 60,
    sanitizer: str = "none",
) -> DynamicRunResult:
    cli = _modal_cli(repo_root)
    cmd = [
        sys.executable,
        str(cli),
        "-t",
        task,
        "-m",
        mode,
        str(submission),
        "-o",
        str(output_path),
        "--sanitizer",
        sanitizer,
    ]
    timed_out = False
    returncode = 1
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), check=False, timeout=timeout_seconds)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
    return DynamicRunResult(
        label=f"{mode}:{sanitizer}",
        command=cmd,
        returncode=returncode,
        timed_out=timed_out,
        output_path=output_path,
    )


def run_dynamic_suite(
    *,
    repo_root: Path,
    task: str,
    mode: str,
    submission: Path,
    output_dir: Path,
    timeout_seconds: int = 60,
    include_memcheck: bool = False,
    include_racecheck: bool = False,
) -> list[DynamicRunResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, str]] = [(mode, "none")]
    if include_memcheck:
        runs.append((mode, "memcheck"))
    if include_racecheck:
        runs.append((mode, "racecheck"))

    results: list[DynamicRunResult] = []
    for idx, (run_mode, sanitizer) in enumerate(runs):
        suffix = sanitizer if sanitizer != "none" else "baseline"
        output_path = output_dir / f"dynamic_{idx:02d}_{run_mode}_{suffix}.txt"
        results.append(
            run_modal_eval(
                repo_root=repo_root,
                task=task,
                mode=run_mode,
                submission=submission,
                output_path=output_path,
                timeout_seconds=timeout_seconds,
                sanitizer=sanitizer,
            )
        )
    return results
