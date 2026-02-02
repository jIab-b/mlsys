from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent / "ledger.json"


DEFAULT_TEMPLATES = {
    "gemm_nvfp4_tma_warpspecialized_v4": {
        "id": "gemm_nvfp4_tma_warpspecialized_v4",
        "op_type": "gemm",
        "tags": ["nvfp4", "tma", "tcgen05", "warpspecialized", "split_k"],
        "summary": "B200 nvfp4 block-scaled GEMM with TMA pipeline + tcgen05 MMA",
        "required_ops": [
            "mbarrier_init",
            "mbarrier_arrive_expect_tx_cta",
            "mbarrier_wait",
            "tma_2d_gmem2smem",
            "tcgen05_alloc",
            "tcgen05_cp_nvfp4",
            "tcgen05_mma_nvfp4",
            "tcgen05_commit",
            "tcgen05_wait_ld",
            "tcgen05_dealloc",
        ],
        "required_buffers": ["tmem0", "A_smem", "B_smem", "SFA_smem", "SFB_smem"],
        "params": [
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "NUM_STAGES",
            "SPLIT_K",
            "SWAP_AB",
            "C_N_MAJOR",
        ],
        "constraints": [
            "K % 256 == 0",
            "BLOCK_K == 256",
            "BLOCK_M % 32 == 0",
            "BLOCK_N in [64, 128]",
        ],
        "raw_groups": ["prologue_tma_setup", "mainloop_tma_mma", "epilogue_writeback"],
        "version": 1,
    }
}


def _load(path: Path = DEFAULT_LEDGER_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"templates": dict(DEFAULT_TEMPLATES), "kernels": {}, "runs": []}
    return json.loads(path.read_text())


def _save(data: Dict[str, Any], path: Path = DEFAULT_LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def reset_ledger(path: Path = DEFAULT_LEDGER_PATH) -> None:
    _save({"templates": dict(DEFAULT_TEMPLATES), "kernels": {}, "runs": []}, path)


def upsert_template(template: Dict[str, Any], path: Path = DEFAULT_LEDGER_PATH) -> None:
    data = _load(path)
    templates = data.setdefault("templates", {})
    templates[str(template.get("id"))] = template
    _save(data, path)


def insert_kernel(kernel: Dict[str, Any], path: Path = DEFAULT_LEDGER_PATH) -> None:
    data = _load(path)
    kernels = data.setdefault("kernels", {})
    kernels[str(kernel.get("id"))] = kernel
    _save(data, path)


def insert_run(run: Dict[str, Any], path: Path = DEFAULT_LEDGER_PATH) -> None:
    data = _load(path)
    runs = data.setdefault("runs", [])
    runs.append(run)
    _save(data, path)


def load_templates(op_type: Optional[str] = None, path: Path = DEFAULT_LEDGER_PATH) -> Iterable[Dict[str, Any]]:
    data = _load(path)
    for tmpl in data.get("templates", {}).values():
        if op_type is None or tmpl.get("op_type") == op_type:
            yield tmpl


def best_runs(
    op_type: str,
    limit: int = 5,
    path: Path = DEFAULT_LEDGER_PATH,
) -> Iterable[Dict[str, Any]]:
    data = _load(path)
    templates = data.get("templates", {})
    kernels = data.get("kernels", {})
    runs = data.get("runs", [])
    allowed_templates = {tid for tid, t in templates.items() if t.get("op_type") == op_type}
    allowed_kernels = {kid for kid, k in kernels.items() if k.get("template_id") in allowed_templates}
    filtered = [r for r in runs if r.get("kernel_id") in allowed_kernels and r.get("status") == "success"]
    filtered.sort(key=lambda r: r.get("time_us") if r.get("time_us") is not None else float("inf"))
    for row in filtered[:limit]:
        yield row
