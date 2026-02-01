"""Reference implementation for gemm_n128_k2048.
Auto-extracted from flashinfer-bench definitions.
Op type: gemm
"""
import torch

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
