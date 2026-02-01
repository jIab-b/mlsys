"""Auto-generated registry of reference implementations."""
from pathlib import Path
import importlib.util

_ROOT = Path(__file__).parent

REFS = {
    "dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps1": {"op_type": "dsa_paged", "file": "torch/dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps1.py"},
    "dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64": {"op_type": "dsa_paged", "file": "torch/dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64.py"},
    "dsa_topk_indexer_fp8_h64_d128_topk256_ps64": {"op_type": "dsa_paged", "file": "torch/dsa_topk_indexer_fp8_h64_d128_topk256_ps64.py"},
    "fused_add_rmsnorm_h2048": {"op_type": "rmsnorm", "file": "torch/fused_add_rmsnorm_h2048.py"},
    "fused_add_rmsnorm_h4096": {"op_type": "rmsnorm", "file": "torch/fused_add_rmsnorm_h4096.py"},
    "fused_add_rmsnorm_h7168": {"op_type": "rmsnorm", "file": "torch/fused_add_rmsnorm_h7168.py"},
    "gdn_decode_qk16_v32_d128_k_last": {"op_type": "gdn", "file": "torch/gdn_decode_qk16_v32_d128_k_last.py"},
    "gdn_prefill_qk16_v32_d128_k_last": {"op_type": "gdn", "file": "torch/gdn_prefill_qk16_v32_d128_k_last.py"},
    "gemm_n128_k2048": {"op_type": "gemm", "file": "torch/gemm_n128_k2048.py"},
    "gemm_n2048_k4096": {"op_type": "gemm", "file": "torch/gemm_n2048_k4096.py"},
    "gemm_n256_k7168": {"op_type": "gemm", "file": "torch/gemm_n256_k7168.py"},
    "gemm_n28672_k4096": {"op_type": "gemm", "file": "torch/gemm_n28672_k4096.py"},
    "gemm_n4096_k14336": {"op_type": "gemm", "file": "torch/gemm_n4096_k14336.py"},
    "gemm_n4096_k4096": {"op_type": "gemm", "file": "torch/gemm_n4096_k4096.py"},
    "gemm_n5120_k2048": {"op_type": "gemm", "file": "torch/gemm_n5120_k2048.py"},
    "gemm_n6144_k4096": {"op_type": "gemm", "file": "torch/gemm_n6144_k4096.py"},
    "gqa_paged_decode_h32_kv4_d128_ps1": {"op_type": "gqa_paged", "file": "torch/gqa_paged_decode_h32_kv4_d128_ps1.py"},
    "gqa_paged_decode_h32_kv4_d128_ps64": {"op_type": "gqa_paged", "file": "torch/gqa_paged_decode_h32_kv4_d128_ps64.py"},
    "gqa_paged_decode_h32_kv8_d128_ps1": {"op_type": "gqa_paged", "file": "torch/gqa_paged_decode_h32_kv8_d128_ps1.py"},
    "gqa_paged_decode_h32_kv8_d128_ps64": {"op_type": "gqa_paged", "file": "torch/gqa_paged_decode_h32_kv8_d128_ps64.py"},
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": {"op_type": "gqa_paged", "file": "torch/gqa_paged_prefill_causal_h32_kv4_d128_ps1.py"},
    "gqa_paged_prefill_causal_h32_kv4_d128_ps64": {"op_type": "gqa_paged", "file": "torch/gqa_paged_prefill_causal_h32_kv4_d128_ps64.py"},
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": {"op_type": "gqa_paged", "file": "torch/gqa_paged_prefill_causal_h32_kv8_d128_ps1.py"},
    "gqa_paged_prefill_causal_h32_kv8_d128_ps64": {"op_type": "gqa_paged", "file": "torch/gqa_paged_prefill_causal_h32_kv8_d128_ps64.py"},
    "gqa_ragged_prefill_causal_h32_kv4_d128": {"op_type": "gqa_ragged", "file": "torch/gqa_ragged_prefill_causal_h32_kv4_d128.py"},
    "gqa_ragged_prefill_causal_h32_kv8_d128": {"op_type": "gqa_ragged", "file": "torch/gqa_ragged_prefill_causal_h32_kv8_d128.py"},
    "mla_paged_decode_h16_ckv512_kpe64_ps1": {"op_type": "mla_paged", "file": "torch/mla_paged_decode_h16_ckv512_kpe64_ps1.py"},
    "mla_paged_decode_h16_ckv512_kpe64_ps64": {"op_type": "mla_paged", "file": "torch/mla_paged_decode_h16_ckv512_kpe64_ps64.py"},
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": {"op_type": "mla_paged", "file": "torch/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.py"},
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps64": {"op_type": "mla_paged", "file": "torch/mla_paged_prefill_causal_h16_ckv512_kpe64_ps64.py"},
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": {"op_type": "moe", "file": "torch/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py"},
    "rmsnorm_h128": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h128.py"},
    "rmsnorm_h1536": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h1536.py"},
    "rmsnorm_h2048": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h2048.py"},
    "rmsnorm_h4096": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h4096.py"},
    "rmsnorm_h512": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h512.py"},
    "rmsnorm_h7168": {"op_type": "rmsnorm", "file": "torch/rmsnorm_h7168.py"},
    "top_k_sampling_from_probs_v128256": {"op_type": "sampling", "file": "torch/top_k_sampling_from_probs_v128256.py"},
    "top_k_sampling_from_probs_v129280": {"op_type": "sampling", "file": "torch/top_k_sampling_from_probs_v129280.py"},
    "top_k_sampling_from_probs_v151936": {"op_type": "sampling", "file": "torch/top_k_sampling_from_probs_v151936.py"},
    "top_k_top_p_sampling_from_probs_v128256": {"op_type": "sampling", "file": "torch/top_k_top_p_sampling_from_probs_v128256.py"},
    "top_k_top_p_sampling_from_probs_v129280": {"op_type": "sampling", "file": "torch/top_k_top_p_sampling_from_probs_v129280.py"},
    "top_k_top_p_sampling_from_probs_v151936": {"op_type": "sampling", "file": "torch/top_k_top_p_sampling_from_probs_v151936.py"},
    "top_p_sampling_from_probs_v128256": {"op_type": "sampling", "file": "torch/top_p_sampling_from_probs_v128256.py"},
    "top_p_sampling_from_probs_v129280": {"op_type": "sampling", "file": "torch/top_p_sampling_from_probs_v129280.py"},
    "top_p_sampling_from_probs_v151936": {"op_type": "sampling", "file": "torch/top_p_sampling_from_probs_v151936.py"},
}

def load_ref(name: str):
    """Load a reference implementation by name."""
    if name not in REFS:
        raise KeyError(f"Unknown reference: {name}")
    
    file_path = _ROOT / REFS[name]["file"]
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run
