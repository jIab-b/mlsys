"""Modal infrastructure for eval_suite (sparse attention) on B200."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import modal

# --- Configuration ---
APP_NAME = "kernel_app"
VOLUME_NAME = "kernel_vol"
#BASE_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20260120-09a9d214"
BASE_IMAGE = "lmsysorg/sglang:nightly-dev-20260126-48f4340b"
IMAGE_ENV = {
    "HF_HOME": "/kernel_data/hf",
    "HUGGINGFACE_HUB_CACHE": "/kernel_data/hf",
}

GPU_DEFAULT = "B200"
GPU_ALIASES = {
    "L4": "L4",
    "L40S": "L40S",
    "A100": "A100",
    "H100": "H100",
    "B200": "B200",
}

# --- Paths ---
LOCAL_EVAL_SUITE = Path(__file__).parent.parent  # eval_suite/
PROJECT_ROOT = LOCAL_EVAL_SUITE.parent  # mlsys/
LOCAL_OUTPUTS = LOCAL_EVAL_SUITE / "modal" / "outputs"
VOLUME_MOUNT_PATH = PurePosixPath("/kernel_data")
CONTAINER_EVAL_SUITE = VOLUME_MOUNT_PATH / "eval_suite"
VOLUME_EVAL_SUITE = PurePosixPath("/eval_suite")
REMOTE_OUTPUTS = PurePosixPath("/outputs")  # Volume path for output sync
SGLANG_MANIFEST_PATH = "/manifest.json"

# Compile-cache paths (system paths are ephemeral; volume paths are persistent).
SYS_TORCH_EXTENSIONS_DIR = Path("/root/.cache/torch_extensions")
SYS_CUDA_COMPUTE_CACHE_DIR = Path("/root/.nv/ComputeCache")
VOL_CACHE_ROOT = Path("/kernel_data/kernel_cache")
VOL_TORCH_EXTENSIONS_DIR = VOL_CACHE_ROOT / "torch_extensions"
VOL_CUDA_COMPUTE_CACHE_DIR = VOL_CACHE_ROOT / "cuda_compute"


ADDITIONAL_DEPS = [
    "transformers==5.0.0",
    "safetensors",
]


LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "zai-org/GLM-4.7-Flash")


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def _count_so_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*.so") if p.is_file())


def _copy_tree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def _restore_compile_cache_from_volume() -> dict[str, int]:
    VOL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_TORCH_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    VOL_CUDA_COMPUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _copy_tree_replace(VOL_TORCH_EXTENSIONS_DIR, SYS_TORCH_EXTENSIONS_DIR)
    _copy_tree_replace(VOL_CUDA_COMPUTE_CACHE_DIR, SYS_CUDA_COMPUTE_CACHE_DIR)

    return {
        "torch_files": _count_files(SYS_TORCH_EXTENSIONS_DIR),
        "torch_so_files": _count_so_files(SYS_TORCH_EXTENSIONS_DIR),
        "cuda_files": _count_files(SYS_CUDA_COMPUTE_CACHE_DIR),
    }


def _save_compile_cache_to_volume() -> dict[str, int]:
    VOL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_TORCH_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    VOL_CUDA_COMPUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _copy_tree_replace(SYS_TORCH_EXTENSIONS_DIR, VOL_TORCH_EXTENSIONS_DIR)
    _copy_tree_replace(SYS_CUDA_COMPUTE_CACHE_DIR, VOL_CUDA_COMPUTE_CACHE_DIR)

    return {
        "torch_files": _count_files(VOL_TORCH_EXTENSIONS_DIR),
        "torch_so_files": _count_so_files(VOL_TORCH_EXTENSIONS_DIR),
        "cuda_files": _count_files(VOL_CUDA_COMPUTE_CACHE_DIR),
    }


def _build_image() -> modal.Image:
    """Build the container image with sglang + Blender dependencies."""
    image = (
        modal.Image.from_registry(BASE_IMAGE)
        .env(IMAGE_ENV)
        .uv_pip_install(*ADDITIONAL_DEPS)
    )
    return image


def _gpu_type(name: str | None = None) -> str:
    alias = name or GPU_DEFAULT
    return GPU_ALIASES.get(alias, alias)


def _cpu_model_name() -> str:
    import platform
    cpu = platform.processor()
    if cpu:
        return cpu
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Unknown"


def _system_info() -> dict[str, str | int]:
    """Collect system information."""
    import platform
    import torch

    has_cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if has_cuda and torch.cuda.device_count() > 0 else "CPU"
    device_count = torch.cuda.device_count() if has_cuda else 0

    return {
        "gpu": gpu,
        "cpu": _cpu_model_name(),
        "device_count": device_count,
        "runtime": "CUDA" if has_cuda else "CPU",
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "hostname": platform.node(),
    }


# --- Modal App ---
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = _build_image()


# --- Hash-based sync ---
def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _get_local_manifest(base_path: Path, extensions: set[str] | None = None) -> dict[str, str]:
    """Build manifest with hashes for all files to sync."""
    manifest = {}
    if not base_path.exists():
        return manifest

    for f in base_path.rglob("*"):
        if not f.is_file():
            continue
        # Skip common non-essential files
        if any(skip in f.parts for skip in ["__pycache__", ".git", ".pytest_cache", "*.egg-info", "build", "dist"]):
            continue
        if f.name.endswith(":Zone.Identifier"):
            continue
        if extensions and f.suffix not in extensions:
            continue

        rel_path = f.relative_to(base_path)
        manifest[str(rel_path)] = _file_hash(f)

    return manifest


def _get_remote_manifest(manifest_path: str) -> dict[str, str]:
    try:
        data = b"".join(volume.read_file(manifest_path))
        return json.loads(data)
    except Exception:
        return {}


def _sync_directory(
    local_root: Path,
    remote_root: PurePosixPath,
    manifest_path: str,
    extensions: set[str] | None = None,
    force: bool = False,
) -> int:
    """Sync a local directory tree to the Modal volume."""
    if not local_root.exists():
        print(f"Local path not found: {local_root}")
        return 0

    print(f"Scanning {local_root}...")
    local = _get_local_manifest(local_root, extensions=extensions)
    remote = {} if force else _get_remote_manifest(manifest_path)

    changed = [k for k, v in local.items() if remote.get(k) != v]
    deleted = [k for k in remote if k not in local]

    if not changed and not deleted:
        print("No changes to sync.")
        return 0

    print(f"Uploading {len(changed)} changed file(s)...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(local, f)
        manifest_tmp = f.name

    with volume.batch_upload(force=True) as batch:
        for key in changed:
            local_path = local_root / key
            remote_path = str(remote_root / key)
            batch.put_file(str(local_path), remote_path)
        batch.put_file(manifest_tmp, manifest_path)

    Path(manifest_tmp).unlink()
    print(f"Synced {len(changed)} file(s), {len(deleted)} removed from tracking.")
    return len(changed)


SYNC_EXTENSIONS = {".py", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".toml", ".safetensors"}


def sync_sglang() -> int:
    """Sync eval_suite directory to Modal volume."""
    changed = _sync_directory(
        local_root=LOCAL_EVAL_SUITE,
        remote_root=VOLUME_EVAL_SUITE,
        manifest_path=SGLANG_MANIFEST_PATH,
        extensions=SYNC_EXTENSIONS,
    )
    if changed == 0:
        # Check critical paths exist
        critical_paths = [
            str(VOLUME_EVAL_SUITE / "test_kernels" / "sparse_attention" / "eval.py"),
            str(VOLUME_EVAL_SUITE / "common" / "workload_loader.py"),
        ]
        needs_sync = False
        for path in critical_paths:
            try:
                list(volume.read_file(path))
            except Exception:
                needs_sync = True
                break
        if needs_sync:
            changed = _sync_directory(
                local_root=LOCAL_EVAL_SUITE,
                remote_root=VOLUME_EVAL_SUITE,
                manifest_path=SGLANG_MANIFEST_PATH,
                extensions=SYNC_EXTENSIONS,
                force=True,
            )
    return changed


def sync_mcp() -> int:
    """Sync MCP-related directories to Modal volume."""
    return 0


# --- Output sync ---
def download_file(remote_path: str, local_path: Path) -> bool:
    """Download a file from Modal volume to local path."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as fh:
            for chunk in volume.read_file(remote_path):
                fh.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {remote_path}: {e}")
        return False


def delete_remote_file(remote_path: str) -> bool:
    """Delete a file from Modal volume."""
    try:
        volume.remove_file(remote_path)
        return True
    except Exception:
        return False


def sync_outputs(clean: bool = True) -> int:
    """Download all files from /outputs on volume to local outputs dir."""
    try:
        entries = volume.listdir(str(REMOTE_OUTPUTS), recursive=True)
    except FileNotFoundError:
        print(f"No outputs at {REMOTE_OUTPUTS}")
        return 0

    if not entries:
        print(f"No outputs at {REMOTE_OUTPUTS}")
        return 0

    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue

        remote_path = entry.path
        rel_path = PurePosixPath(remote_path).relative_to(REMOTE_OUTPUTS)
        local_target = LOCAL_OUTPUTS / Path(str(rel_path))

        if download_file(remote_path, local_target):
            downloaded += 1
            if clean:
                delete_remote_file(remote_path)

    print(f"Downloaded {downloaded} file(s) to {LOCAL_OUTPUTS}")
    return downloaded


# Map task names to their directory paths
TASK_PATHS = {
    "sparse_attention": "test_kernels/sparse_attention",
    "sparse_index": "test_kernels/sparse_index",
    "sparse_attn": "test_kernels/sparse_attn",
    "dsa_index_2048": "test_kernels/dsa_index_2048",
    "dsa_attn_2048": "test_kernels/dsa_attn_2048",
}


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT_PATH): volume},
    gpu=_gpu_type(),
    timeout=1200,
)
def run_eval(
    submission_code: str,
    tests_content: str,
    mode: str = "test",
    trace_dir: str | None = None,
    definition_name: str | None = None,
    solution_name: str | None = None,
    op_type: str | None = None,
    task: str = "sparse_attention",
) -> dict:
    """Run eval_suite task eval remotely."""
    import sys

    cache_sync = {"restore": {}, "save": {}, "errors": []}

    try:
        try:
            cache_sync["restore"] = _restore_compile_cache_from_volume()
        except Exception as exc:
            cache_sync["errors"].append(f"restore_failed: {exc}")

        task_path = TASK_PATHS.get(task, f"test_kernels/{task}")
        work = Path(CONTAINER_EVAL_SUITE / task_path)
        work.mkdir(parents=True, exist_ok=True)

        (work / "submission.py").write_text(submission_code)
        (work / "tests.txt").write_text(tests_content)

        r, w = os.pipe()
        os.set_inheritable(w, True)
        env = os.environ.copy()
        env["POPCORN_FD"] = str(w)
        if definition_name:
            env["TRACE_DEFINITION"] = definition_name
        if solution_name:
            env["TRACE_SOLUTION"] = solution_name
        if op_type:
            env["TRACE_OP_TYPE"] = op_type
        if trace_dir:
            env["TRACE_OUT"] = trace_dir
            Path(trace_dir).mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            [sys.executable, "eval.py", mode, "tests.txt"],
            cwd=str(work),
            env=env,
            pass_fds=(w,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.close(w)

        stdout, stderr = proc.communicate()
        output = os.read(r, 1 << 20).decode()
        os.close(r)

        result = {
            "popcorn": output,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "mode": mode,
            "system": _system_info(),
        }
        if trace_dir:
            import base64
            import tarfile
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with tarfile.open(tmp_path, "w:gz") as tar:
                    tar.add(trace_dir, arcname=".")
                payload = tmp_path.read_bytes()
                result["trace_archive_b64"] = base64.b64encode(payload).decode("ascii")
                result["trace_archive_name"] = "trace.tar.gz"
            finally:
                tmp_path.unlink(missing_ok=True)
                shutil.rmtree(trace_dir, ignore_errors=True)

        result["cache_sync"] = cache_sync
        return result
    finally:
        try:
            cache_sync["save"] = _save_compile_cache_to_volume()
        except Exception as exc:
            cache_sync["errors"].append(f"save_failed: {exc}")


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT_PATH): volume},
    gpu=_gpu_type(),
    timeout=600,
)
def run_kernel_test(kernel_code: str, kernel_name: str, mode: str = "test") -> dict:
    """Run a kernel file for testing or benchmarking against FlashInfer.

    mode: "test" runs test_correctness(), "bench" runs benchmark()
    """
    import sys
    import importlib.util
    import time
    import torch

    work = Path("/tmp/kernel_test")
    work.mkdir(parents=True, exist_ok=True)

    kernel_file = work / f"{kernel_name}.py"
    kernel_file.write_text(kernel_code)

    # Load the module
    spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
    module = importlib.util.module_from_spec(spec)

    result = {
        "system": _system_info(),
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }

    try:
        spec.loader.exec_module(module)

        if mode == "bench":
            # Run benchmark
            import inspect
            benchmarks = []
            generate_fn = getattr(module, "generate_random_inputs", None)
            run_fn = getattr(module, "run", None)

            if not generate_fn or not run_fn:
                result["returncode"] = 1
                result["stderr"] = "Missing generate_random_inputs() or run() function"
                return result

            # Auto-detect required params from generate_random_inputs signature
            sig = inspect.signature(generate_fn)
            params = list(sig.parameters.keys())

            # Build configs based on what params are needed
            if "max_seq_len" in params:
                # Attention-style kernel (MLA, GQA, etc.)
                configs = [
                    {"batch_size": 1, "max_seq_len": 64, "name": "batch=1,seq=64"},
                    {"batch_size": 8, "max_seq_len": 128, "name": "batch=8,seq=128"},
                    {"batch_size": 32, "max_seq_len": 256, "name": "batch=32,seq=256"},
                ]
            elif "with_residual" in params:
                # RMSNorm-style kernel
                configs = [
                    {"batch_size": 1, "name": "batch=1"},
                    {"batch_size": 8, "name": "batch=8"},
                    {"batch_size": 32, "name": "batch=32"},
                ]
            else:
                # Generic fallback
                configs = [
                    {"batch_size": 1, "name": "batch=1"},
                    {"batch_size": 8, "name": "batch=8"},
                    {"batch_size": 32, "name": "batch=32"},
                ]

            # Get run() signature to filter inputs
            run_sig = inspect.signature(run_fn)
            run_params = set(run_sig.parameters.keys())

            for cfg in configs:
                try:
                    # Only pass params that generate_fn accepts
                    valid_cfg = {k: v for k, v in cfg.items() if k in params}
                    inputs = generate_fn(**valid_cfg)

                    # Filter inputs to only what run() accepts
                    if isinstance(inputs, dict):
                        run_inputs = {k: v for k, v in inputs.items() if k in run_params}
                    else:
                        run_inputs = inputs

                    # Warmup
                    for _ in range(10):
                        if isinstance(run_inputs, dict):
                            run_fn(**run_inputs)
                        else:
                            run_fn(*run_inputs)
                    torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    for _ in range(100):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if isinstance(run_inputs, dict):
                            run_fn(**run_inputs)
                        else:
                            run_fn(*run_inputs)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1e6)

                    import statistics
                    benchmarks.append({
                        "name": cfg["name"],
                        "mean_us": statistics.mean(times),
                        "std_us": statistics.stdev(times) if len(times) > 1 else 0,
                        "best_us": min(times),
                        "worst_us": max(times),
                    })
                except Exception as e:
                    benchmarks.append({
                        "name": cfg["name"],
                        "error": str(e),
                    })

            result["benchmarks"] = benchmarks

        else:  # mode == "test"
            # Run test_correctness or main
            test_fn = getattr(module, "test_correctness", None)
            main_fn = getattr(module, "main", None)

            tests = []
            if test_fn:
                # Run single test
                try:
                    passed = test_fn()
                    tests.append({
                        "name": "test_correctness",
                        "status": "pass" if passed else "fail",
                    })
                except Exception as e:
                    tests.append({
                        "name": "test_correctness",
                        "status": "fail",
                        "error": str(e),
                    })
                    result["returncode"] = 1
            elif main_fn:
                # Run main and capture output
                import io
                import contextlib

                stdout_capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout_capture):
                        main_fn()
                    result["stdout"] = stdout_capture.getvalue()
                    # Parse output for pass/fail
                    output = result["stdout"]
                    if "All tests passed" in output or "PASSED" in output:
                        tests.append({"name": "main", "status": "pass"})
                    elif "FAILED" in output or "failed" in output.lower():
                        tests.append({"name": "main", "status": "fail"})
                        result["returncode"] = 1
                    else:
                        tests.append({"name": "main", "status": "pass"})
                except Exception as e:
                    tests.append({
                        "name": "main",
                        "status": "fail",
                        "error": str(e),
                    })
                    result["returncode"] = 1
            else:
                result["returncode"] = 1
                result["stderr"] = "No test_correctness() or main() function found"

            result["tests"] = tests

    except Exception as e:
        import traceback
        result["returncode"] = 1
        result["stderr"] = traceback.format_exc()

    return result


# --- Shell helpers ---
@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume})
def volume_shell() -> None:
    """Open a Modal shell with the volume mounted (no GPU)."""


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type())
def gpu_shell() -> None:
    """Open a Modal shell with volume and GPU attached."""


__all__ = [
    "app",
    "image",
    "volume",
    "sync_sglang",
    "sync_mcp",
    "sync_outputs",
    "download_file",
    "delete_remote_file",
    "LOCAL_OUTPUTS",
    "REMOTE_OUTPUTS",
    "LLM_MODEL_DEFAULT",
    "_system_info",
    "_gpu_type",
    "VOLUME_MOUNT_PATH",
    "volume_shell",
    "gpu_shell",
    "run_eval",
]
