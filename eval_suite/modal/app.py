import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
import modal


APP_NAME = "kernel_app"
VOLUME_NAME = "kernel_vol"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"

IMAGE_ENV = {
    "HF_HOME": "/kernel_data/hf",
    "HUGGINGFACE_HUB_CACHE": "/kernel_data/hf",
}
RUN_COMMANDS = [
    "apt-get update && apt-get install -y curl ca-certificates gnupg",
    "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
    "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
    "apt-get update",
]
APT_PACKAGES = [
    "graphviz",
]
UV_PACKAGES = [
    "nvidia-cutlass-dsl",
    "nvtx",
    "pydot",
]
GPU_DEFAULT = "B200"
GPU_ALIASES = {
    "L4": "L4",
    "L40S": "L40S",
    "A100": "A100",
    "H100": "H100",
    "B200": "B200",
}

VOLUME_MOUNT_PATH = PurePosixPath("/kernel_data")


def _build_image() -> modal.Image:
    image = modal.Image.from_registry(BASE_IMAGE).env(IMAGE_ENV)
   # for command in RUN_COMMANDS:
   #     image = image.run_commands(command)
   # if APT_PACKAGES:
   #     image = image.apt_install(*APT_PACKAGES)
   # image = image.run_commands("pip install --upgrade pip uv")
   # if UV_PACKAGES:
   #     image = image.run_commands("uv pip install --system " + " ".join(UV_PACKAGES))
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


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = _build_image()


# --- Hash-based sync ---
PROJECT_ROOT = Path(__file__).parent.parent  # eval_suite/
MANIFEST_PATH = "/manifest_v2.json"

# Task directories to sync (each becomes /{dir_name}/ on volume)
TASK_DIRS = [
    "common",
    "test_kernels/sparse_attention",
    "test_kernels/sparse_index",
    "test_kernels/sparse_attn",
    "test_kernels/dsa_index_2048",
    "test_kernels/dsa_attn_2048",
]


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _get_sync_mapping() -> dict[str, tuple[Path, str]]:
    """Build mapping of manifest_key -> (local_path, remote_path) for all task files."""
    mapping = {}
    for task_dir in TASK_DIRS:
        task_local_dir = PROJECT_ROOT / task_dir
        if task_local_dir.exists():
            for f in task_local_dir.iterdir():
                if f.is_file() and not f.name.endswith(":Zone.Identifier"):
                    manifest_key = f"{task_dir}/{f.name}"
                    remote_path = f"/{task_dir}/{f.name}"
                    mapping[manifest_key] = (f, remote_path)
    return mapping


def _get_remote_manifest() -> dict[str, str]:
    try:
        data = b"".join(volume.read_file(MANIFEST_PATH))
        return json.loads(data)
    except Exception:
        return {}


def sync_project() -> int:
    """Sync project files to volume, uploading only changed files."""
    mapping = _get_sync_mapping()
    local = {key: _file_hash(local_path) for key, (local_path, _) in mapping.items()}
    remote = _get_remote_manifest()

    changed = [k for k, v in local.items() if remote.get(k) != v]
    deleted = [k for k in remote if k not in local]

    if not changed and not deleted:
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
    print(f"Synced {len(changed)} file(s), {len(deleted)} removed.")
    return len(changed)


# --- Remote eval ---
TASK_PATHS = {
    "sparse_attention": "test_kernels/sparse_attention",
    "sparse_index": "test_kernels/sparse_index",
    "sparse_attn": "test_kernels/sparse_attn",
    "dsa_index_2048": "test_kernels/dsa_index_2048",
    "dsa_attn_2048": "test_kernels/dsa_attn_2048",
}

EVAL_SCRIPTS = {k: "eval.py" for k in TASK_PATHS}

@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, timeout=600)
def vol_shell():
    pass



@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type(), timeout=600)
def run_eval(submission_code: str, tests_content: str, mode: str = "test", workspace_name: str = "sparse_attention") -> dict:
    """Run eval remotely with given submission and tests."""
    import sys

    task_path = TASK_PATHS.get(workspace_name, f"test_kernels/{workspace_name}")
    work = Path(f"{VOLUME_MOUNT_PATH}/{task_path}")
    work.mkdir(parents=True, exist_ok=True)

    (work / "submission.py").write_text(submission_code)
    (work / "tests.txt").write_text(tests_content)

    eval_script = EVAL_SCRIPTS.get(workspace_name, "eval.py")

    r, w = os.pipe()
    os.set_inheritable(w, True)
    env = os.environ.copy()
    env["POPCORN_FD"] = str(w)

    proc = subprocess.Popen(
        [sys.executable, eval_script, mode, "tests.txt"],
        cwd=str(work), env=env, pass_fds=(w,),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
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


__all__ = [
    "app",
    "image",
    "volume",
    "sync_project",
    "run_eval",
]
