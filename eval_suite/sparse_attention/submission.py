from task import input_t, output_t


def compile_kernel():
    """Optional: compile/load PTX/CUDA here."""
    return None


def custom_kernel(data: input_t) -> output_t:
    """
    Implement sparse attention kernel.

    Expected data tuple:
      (q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
       q_nope, q_pe, ckv_cache, kpe_cache, sm_scale)

    Should return (output, lse) matching reference.
    """
    # TODO: replace with kernel implementation
    raise NotImplementedError
