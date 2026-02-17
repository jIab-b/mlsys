import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path

try:
    from tvm.ffi import register_func
except Exception:
    def register_func(_name):
        def deco(fn):
            return fn

        return deco

_module = None
_spill_workspace = {}


def _read_cuda_source() -> str:
    here = Path(__file__).resolve().parent
    for candidate in ("dsa_index.cu", "kernel.cu"):
        path = here / candidate
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"No CUDA source found in {here} (expected dsa_index.cu or kernel.cu)")


def _get_module():
    global _module
    if _module is None:
        cuda_src = _read_cuda_source()
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def compile_kernel() -> None:
    _get_module()


def _get_spill_workspace(device: torch.device, batch: int, spill_stride: int) -> torch.Tensor:
    dev_key = (device.type, -1 if device.index is None else device.index)
    ws = _spill_workspace.get(dev_key)

    if ws is None or ws.shape[0] < batch or ws.shape[1] < spill_stride:
        rows = batch if ws is None else max(batch, ws.shape[0])
        cols = spill_stride if ws is None else max(spill_stride, ws.shape[1])
        ws = torch.empty((rows, cols), dtype=torch.float32, device=device)
        _spill_workspace[dev_key] = ws

    return ws[:batch, :spill_stride]


def dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch = int(q_index_fp8.shape[0])
    spill_stride = int(block_table.shape[1]) * 64
    spill_scores = _get_spill_workspace(q_index_fp8.device, batch, spill_stride)

    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        spill_scores,
        topk_indices,
    )

    return topk_indices


@register_func("flashinfer.kernel")
def kernel(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    topk_indices,
):
    return dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
