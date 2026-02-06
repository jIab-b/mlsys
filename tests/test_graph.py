#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_shell(cmd: str, *, cwd: Path) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=str(cwd), check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    out_py = tests_dir / "sub_gemm1.py"
    out_txt = tests_dir / "out.txt"
    venv_activate = tests_dir / "venv" / "bin" / "activate"
    modal_cli = tests_dir / "eval_test" / "modal" / "cli.py"

    if not venv_activate.exists():
        raise FileNotFoundError(f"Missing venv activate script: {venv_activate}")
    if not modal_cli.exists():
        raise FileNotFoundError(f"Missing Modal CLI: {modal_cli}")

    # Validate typed graph IR coverage for both gemm1 and grouped_gemm.
    _run(["python", "tests/test_graph_ir.py"], cwd=repo_root)

    # Build sub_test.py from the current graph.
    _run([
        "python",
        "graph/compiler.py",
        "--out",
        "../tests/sub_gemm1.py",
    ], cwd=repo_root)

    # Run the Modal benchmark for the GEMM task inside the tests venv.
    cmd = (
        f"source {venv_activate} && "
        f"python {modal_cli} "
        f"-t gemm -m benchmark {out_py} -o {out_txt}"
    )
    _run_shell(cmd, cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
