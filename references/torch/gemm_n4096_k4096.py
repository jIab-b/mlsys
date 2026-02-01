"""Reference implementation for gemm_n4096_k4096.
Auto-extracted from flashinfer-bench definitions.
Op type: gemm
"""
import torch

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
