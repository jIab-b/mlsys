#!/usr/bin/env python3
"""Single-file Modal eval runner with CPU-precompile + GPU-run split.

Usage:
    python app_cpu.py <submission> -t <task> -m <mode> [-o out.txt] [--no-sync] [--suppress]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal

# ---- Constants ----
APP_NAME = "kernel_app"
VOLUME_NAME = "kernel_vol"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
GPU_DEFAULT = "B200"
VOLUME_MOUNT_PATH = PurePosixPath("/kernel_data")
EXT_CACHE_VOL = f"{VOLUME_MOUNT_PATH}/.torch_ext_cache"
EXT_CACHE_LOCAL = "/tmp/torch_ext"

TASK_DIRS = [
    "common",
    "test_kernels/sparse_attention",
    "test_kernels/sparse_index",
    "test_kernels/sparse_attn",
    "test_kernels/dsa_index_2048",
    "test_kernels/dsa_index_fused",
    "test_kernels/dsa_attn_2048",
]
TASK_PATHS = {
    "sparse_attention": "test_kernels/sparse_attention",
    "sparse_index": "test_kernels/sparse_index",
    "sparse_attn": "test_kernels/sparse_attn",
    "dsa_index_2048": "test_kernels/dsa_index_2048",
    "dsa_index_fused": "test_kernels/dsa_index_fused",
    "dsa_attn_2048": "test_kernels/dsa_attn_2048",
}
EVAL_SCRIPTS = {k: "eval.py" for k in TASK_PATHS}
TASKS = {k: k for k in TASK_PATHS}

# ---- Modal setup ----
image = modal.Image.from_registry(BASE_IMAGE).env({
    "HF_HOME": "/kernel_data/hf",
    "HUGGINGFACE_HUB_CACHE": "/kernel_data/hf",
}).run_commands(
    "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && "
    "echo /usr/local/cuda/lib64/stubs > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig"
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_volumes = {str(VOLUME_MOUNT_PATH): volume}

# ---- Sync ----
PROJECT_ROOT = Path(__file__).parent.parent  # eval_suite/
MANIFEST_PATH = "/manifest_v2.json"


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _get_sync_mapping() -> dict[str, tuple[Path, str]]:
    mapping = {}
    for task_dir in TASK_DIRS:
        task_local_dir = PROJECT_ROOT / task_dir
        if task_local_dir.exists():
            for f in task_local_dir.iterdir():
                if f.is_file() and not f.name.endswith(":Zone.Identifier"):
                    manifest_key = f"{task_dir}/{f.name}"
                    mapping[manifest_key] = (f, f"/{task_dir}/{f.name}")
    return mapping


def sync_project() -> int:
    mapping = _get_sync_mapping()
    local = {key: _file_hash(lp) for key, (lp, _) in mapping.items()}
    try:
        remote = json.loads(b"".join(volume.read_file(MANIFEST_PATH)))
    except Exception:
        remote = {}

    changed = [k for k, v in local.items() if remote.get(k) != v]
    if not changed:
        print("No changes to sync.")
        return 0

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(local, f)
        manifest_tmp = f.name

    with volume.batch_upload(force=True) as batch:
        for key in changed:
            local_path, remote_path = mapping[key]
            batch.put_file(str(local_path), remote_path)
        batch.put_file(manifest_tmp, MANIFEST_PATH)

    Path(manifest_tmp).unlink()
    print(f"Synced {len(changed)} file(s).")
    return len(changed)


# ---- Remote: CPU compile ----
@app.function(image=image, volumes=_volumes, timeout=600)
def compile_kernel_remote(submission_code: str, workspace_name: str) -> dict:
    """Compile kernel on CPU-only container, stash artifacts to volume."""
    import sys as _sys

    task_path = TASK_PATHS.get(workspace_name, f"test_kernels/{workspace_name}")
    work = Path(f"{VOLUME_MOUNT_PATH}/{task_path}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(submission_code)

    # Clean old artifacts from volume before compiling
    if os.path.exists(EXT_CACHE_VOL):
        shutil.rmtree(EXT_CACHE_VOL)

    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = EXT_CACHE_LOCAL

    proc = subprocess.Popen(
        [_sys.executable, "-c",
         "import sys; sys.path.insert(0,'.'); from submission import compile_kernel; compile_kernel()"],
        cwd=str(work), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode == 0 and os.path.exists(EXT_CACHE_LOCAL):
        shutil.copytree(EXT_CACHE_LOCAL, EXT_CACHE_VOL)
    volume.commit()

    return {
        "ok": proc.returncode == 0,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


# ---- Remote: GPU eval ----
def _system_info() -> dict:
    import platform
    import torch
    has_cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if has_cuda and torch.cuda.device_count() > 0 else "CPU"
    return {
        "gpu": gpu,
        "cpu": platform.processor() or "Unknown",
        "device_count": torch.cuda.device_count() if has_cuda else 0,
        "runtime": "CUDA" if has_cuda else "CPU",
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "hostname": platform.node(),
    }


@app.function(image=image, volumes=_volumes, gpu=GPU_DEFAULT, timeout=600)
def run_eval(submission_code: str, tests_content: str, mode: str = "test", workspace_name: str = "sparse_attention") -> dict:
    """Run eval on GPU container, using pre-compiled artifacts from volume."""
    import sys as _sys

    task_path = TASK_PATHS.get(workspace_name, f"test_kernels/{workspace_name}")
    work = Path(f"{VOLUME_MOUNT_PATH}/{task_path}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(submission_code)
    (work / "tests.txt").write_text(tests_content)

    # Restore cached build artifacts from volume (read-only, no vol writes)
    import glob
    if os.path.exists(EXT_CACHE_VOL):
        if os.path.exists(EXT_CACHE_LOCAL):
            shutil.rmtree(EXT_CACHE_LOCAL)
        shutil.copytree(EXT_CACHE_VOL, EXT_CACHE_LOCAL)

    # Fail fast if no pre-compiled .so — don't let load_inline recompile on GPU
    so_files = glob.glob(f"{EXT_CACHE_LOCAL}/**/*.so", recursive=True)
    if not so_files:
        return {
            "popcorn": "compile: fail\ncompile.error: No pre-compiled artifacts found\ncheck: fail",
            "stdout": "", "stderr": "No .so found in cache — CPU compile may have failed",
            "mode": mode, "system": _system_info(),
        }

    eval_script = EVAL_SCRIPTS.get(workspace_name, "eval.py")
    r, w = os.pipe()
    os.set_inheritable(w, True)
    env = os.environ.copy()
    env["POPCORN_FD"] = str(w)
    env["TORCH_EXTENSIONS_DIR"] = EXT_CACHE_LOCAL

    proc = subprocess.Popen(
        [_sys.executable, eval_script, mode, "tests.txt", "--no-compile", "--prebuilt-so", EXT_CACHE_LOCAL],
        cwd=str(work), env=env, pass_fds=(w,),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.close(w)
    stdout, stderr = proc.communicate()
    output = os.read(r, 1 << 20).decode()
    os.close(r)

    return {
        "popcorn": output,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "mode": mode,
        "system": _system_info(),
    }


# ---- Output formatting ----
CHECK, CROSS = "\u2705", "\u274c"
STOPWATCH, ZAP, SNAIL, MICRO = "\u23f1", "\u26a1", "\U0001F40C", "\u00b5"


def _reorder_spec(spec: str) -> str:
    kv = {}
    for part in spec.split(";"):
        key, sep, value = part.strip().partition(":")
        if sep:
            kv[key.strip()] = value.strip()
    if not kv:
        return spec.strip()
    order = ["batch", "num_pages", "seq_len", "seed"]
    ordered = [f"{k}: {kv[k]}" for k in order if k in kv]
    ordered.extend(f"{k}: {kv[k]}" for k in sorted(kv) if k not in order)
    return "; ".join(ordered)


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_popcorn_log(log_text: str) -> dict:
    data: dict = {"benchmarks": {}, "tests": {}}
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()

        if key == "compile":
            data["compile"] = value
        elif key == "compile.error":
            data["compile_error"] = value
        elif key == "check":
            data["check"] = value
        elif key == "benchmark-count":
            try:
                data["benchmark_count"] = int(value)
            except ValueError:
                data["benchmark_count"] = value
        elif key.startswith("benchmark."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                data["benchmarks"].setdefault(int(parts[1]), {})[parts[2]] = value
        elif key == "test-count":
            try:
                data["test_count"] = int(value)
            except ValueError:
                data["test_count"] = value
        elif key.startswith("test."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                data["tests"].setdefault(int(parts[1]), {})[parts[2]] = value
    return data


def _format_benchmarks(benchmarks: dict) -> list[str]:
    lines = ["## Benchmarks:", "```"]
    items = sorted(benchmarks.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))
        mean, err = _to_float(info.get("mean")), _to_float(info.get("err"))
        best, worst = _to_float(info.get("best")), _to_float(info.get("worst"))
        if None not in (mean, err, best, worst):
            lines.append(f" {STOPWATCH} {mean / 1000:.1f} \u00b1 {err / 1000:.2f} {MICRO}s")
            lines.append(f" {ZAP} {best / 1000:.1f} {MICRO}s {SNAIL} {worst / 1000:.1f} {MICRO}s")
        error = info.get("error")
        if info.get("status") == "fail" or error:
            lines.append(f" {CROSS} {error or 'failed'}")
        if idx != items[-1][0]:
            lines.append("")
    lines.append("```")
    return lines


def _format_tests(tests: dict) -> list[str]:
    lines = ["## Tests:", "```"]
    items = sorted(tests.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))
        status = info.get("status", "")
        if status == "pass":
            lines.append(f" {CHECK} pass")
        elif status == "fail":
            lines.append(f" {CROSS} fail")
        error = info.get("error")
        if error:
            lines.append(f"   {error}")
        if idx != items[-1][0]:
            lines.append("")
    lines.append("```")
    return lines


def _format_output(stdout: str, stderr: str) -> list[str]:
    combined = stdout or ""
    if stderr:
        combined = (combined.rstrip("\n") + "\n--- stderr ---\n" + stderr) if combined else ("--- stderr ---\n" + stderr)
    lines = ["## Program stdout:", "```"]
    if combined:
        lines.extend(combined.rstrip("\n").splitlines())
    lines.append("```")
    return lines


def _build_formatted(data: dict, stdout: str, stderr: str, system: dict | None, mode: str) -> str:
    compile_status = data.get("compile")
    check_status = data.get("check")
    success = check_status == "pass" and compile_status != "fail"

    lines = [f"\"**Modal {CHECK if success else CROSS} {'success' if success else 'failure'}**"]
    if compile_status:
        ok = compile_status == "pass"
        lines.append(f"> {CHECK if ok else CROSS} Compilation {'successful' if ok else 'failed'}")
    if check_status:
        phase = "Benchmarking" if data.get("benchmarks") else "Testing" if data.get("tests") else mode.capitalize()
        ok = check_status == "pass"
        lines.append(f"> {CHECK if ok else CROSS} {phase} {'successful' if ok else 'failed'}")
    lines.append("")

    if system:
        lines.append("Running on:")
        for k, label in [("gpu", "GPU"), ("cpu", "CPU"), ("device_count", "Device count"),
                         ("runtime", "Runtime"), ("platform", "Platform"), ("torch", "Torch"), ("hostname", "Hostname")]:
            if k in system:
                lines.append(f"* {label}: `{system[k]}`")
        lines.extend(["", ""])

    benchmarks, tests = data.get("benchmarks", {}), data.get("tests", {})
    if benchmarks:
        lines.extend(_format_benchmarks(benchmarks))
        lines.append("")
    elif tests:
        lines.extend(_format_tests(tests))
        lines.append("")

    lines.extend(_format_output(stdout, stderr))
    return "\n".join(lines) + "\""


def format_result(result: dict, mode: str) -> str:
    popcorn = result.get("popcorn", "")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    system = result.get("system")

    # Pass through pre-formatted results
    if "## Benchmarks:" in popcorn or "## Program stdout:" in popcorn:
        body = popcorn.strip().strip('"')
        if "## Program stdout:" not in body and (stdout or stderr):
            body = body.rstrip("\n") + "\n\n" + "\n".join(_format_output(stdout, stderr))
        return f"\"{body}\""

    data = _parse_popcorn_log(popcorn)
    return _build_formatted(data, stdout, stderr, system, result.get("mode", mode))


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="CPU-precompile + GPU eval on Modal")
    parser.add_argument("submission", help="Submission file path")
    parser.add_argument("-o", "--output", default="out.txt")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="sparse_attention", choices=list(TASKS.keys()))
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--suppress", dest="suppress_stdout", action="store_true")
    args = parser.parse_args()

    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")

    workspace = TASKS[args.task]
    task_dir = PROJECT_ROOT / "test_kernels" / workspace
    tests_file = task_dir / ("benchmarks.txt" if args.mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    if not args.no_sync:
        sync_project()

    submission_code = submission.read_text()
    tests_content = tests_file.read_text()

    print(f"Running {args.mode} for task '{args.task}' on Modal...", file=sys.stderr)
    with app.run():
        # Step 1: CPU-only compile
        print("Compiling kernel (CPU)...", file=sys.stderr)
        compile_result = compile_kernel_remote.remote(submission_code, workspace)
        if not compile_result["ok"]:
            err = compile_result.get("stderr", "")
            print(f"Compilation failed:\n{err}", file=sys.stderr)
            sys.exit(1)
        print("Compile done. Starting GPU eval...", file=sys.stderr)

        # Step 2: GPU eval
        result = run_eval.remote(submission_code, tests_content, args.mode, workspace)

    formatted = format_result(result, args.mode)
    Path(args.output).write_text(formatted)
    if not args.suppress_stdout:
        print(formatted)
    print(f"Output saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
