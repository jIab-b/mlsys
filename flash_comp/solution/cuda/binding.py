"""Torch-backed binding wrapper for the DSA index kernel."""

from dsa_index import compile_kernel, dsa_topk_indexer


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


__all__ = ["kernel", "compile_kernel"]
