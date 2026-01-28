"""Modal infrastructure for eval_suite (sparse attention) on B200."""
import hashlib
import json
import os
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
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_EVAL_SUITE = PROJECT_ROOT / "eval_suite"
LOCAL_OUTPUTS = PROJECT_ROOT / "modal" / "outputs"
VOLUME_MOUNT_PATH = PurePosixPath("/kernel_data")
CONTAINER_EVAL_SUITE = VOLUME_MOUNT_PATH / "eval_suite"
VOLUME_EVAL_SUITE = PurePosixPath("/eval_suite")
REMOTE_OUTPUTS = PurePosixPath("/outputs")  # Volume path for output sync
SGLANG_MANIFEST_PATH = "/manifest.json"


ADDITIONAL_DEPS = [
    "transformers==5.0.0",
    "safetensors",
]


LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "zai-org/GLM-4.7-Flash")


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
        import shutil
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
