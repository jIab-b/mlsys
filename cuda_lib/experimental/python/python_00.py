# @chunk name=grouped_python_header
#!POPCORN leaderboard nvfp4_grouped_gemm
#!POPCORN gpu NVIDIA
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
# @@LOAD_INLINE@@
# @chunk name=grouped_python_bindings
grouped_gemm_fn = torch.ops.my_grouped_gemm.grouped_gemm

# @chunk name=grouped_python_kernel
def custom_kernel(data: input_t) -> output_t:
    """
    Custom kernel for grouped NVFP4 GEMM.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    
    # Prepare tensor lists
    A_list = []
    B_list = []
    SFA_list = []
    SFB_list = []
    C_list = []
    
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        # a shape: [m, k//2, l=1], squeeze to [m, k//2]
        # b shape: [n, k//2, l=1], squeeze to [n, k//2]
        A_list.append(a.squeeze(-1).contiguous())
        B_list.append(b.squeeze(-1).contiguous())
        C_list.append(c.squeeze(-1).contiguous())
        
        # sfa_reordered shape: [32, 4, rest_m, 4, rest_k, l]
        # Need to permute to [rest_m, rest_k, 32, 4, 4] for kernel
        # Permute: (2, 4, 0, 1, 3, 5) -> [rest_m, rest_k, 32, 4, 4, l]
        # Then squeeze last dim and flatten to contiguous
        sfa_perm = sfa_reordered.permute(2, 4, 0, 1, 3, 5).squeeze(-1).contiguous()
        sfb_perm = sfb_reordered.permute(2, 4, 0, 1, 3, 5).squeeze(-1).contiguous()
        
        # Flatten to 1D for TMA bulk copy (kernel expects [M/128, K/64, 32, 4, 4])
        SFA_list.append(sfa_perm.view(-1))
        SFB_list.append(sfb_perm.view(-1))
    
    # Call the CUDA kernel
    result = grouped_gemm_fn(A_list, B_list, SFA_list, SFB_list, C_list)
    
    # Add back the L dimension
    output = []
    for c in result:
        output.append(c.unsqueeze(-1))
    
    return output
