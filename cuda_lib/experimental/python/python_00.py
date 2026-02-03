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
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        raise RuntimeError("Unexpected input format for grouped GEMM")

    abc_tensors = data[0]

    def _is_pair_list(x):
        if not isinstance(x, (list, tuple)) or len(x) == 0:
            return False
        first = x[0]
        return (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and torch.is_tensor(first[0])
            and torch.is_tensor(first[1])
        )

    candidates = [x for x in data[1:] if _is_pair_list(x) and len(x) == len(abc_tensors)]
    if not candidates:
        raise RuntimeError("Could not locate SFA/SFB tensors in input")

    # Prefer pre-reordered scales (higher dimensional tensors).
    sfasfb_tensors = max(candidates, key=lambda x: x[0][0].dim())

    # Prepare tensor lists
    A_list = []
    B_list = []
    SFA_list = []
    SFB_list = []
    C_list = []

    for (a, b, c), (sfa, sfb) in zip(abc_tensors, sfasfb_tensors):
        # a shape: [m, k//2, l=1], squeeze to [m, k//2]
        # b shape: [n, k//2, l=1], squeeze to [n, k//2]
        A_list.append(a.squeeze(-1).contiguous())
        B_list.append(b.squeeze(-1).contiguous())
        C_list.append(c.squeeze(-1).contiguous())

        def _pack_scales(sf):
            if sf.dim() >= 5:
                # sfa_reordered shape: [32, 4, rest_m, 4, rest_k, l]
                # Permute: (2, 4, 0, 1, 3, 5) -> [rest_m, rest_k, 32, 4, 4, l]
                sf_perm = sf.permute(2, 4, 0, 1, 3, 5).squeeze(-1).contiguous()
            else:
                # Raw shape: [M, K//16, l], reshape to [M/128, K/64, 32, 4, 4]
                sf_raw = sf.squeeze(-1).contiguous()
                m, k16 = sf_raw.shape
                rest_m = m // 128
                rest_k = k16 // 4
                sf_perm = (
                    sf_raw.view(rest_m, 128, rest_k, 4)
                    .view(rest_m, 32, 4, rest_k, 4)
                    .permute(0, 3, 1, 2, 4)
                    .contiguous()
                )
            return sf_perm.view(-1)

        # Flatten to 1D for TMA bulk copy (kernel expects [M/128, K/64, 32, 4, 4])
        SFA_list.append(_pack_scales(sfa))
        SFB_list.append(_pack_scales(sfb))

    # Call the CUDA kernel
    result = grouped_gemm_fn(A_list, B_list, SFA_list, SFB_list, C_list)

    # Add back the L dimension
    output = []
    for c in result:
        output.append(c.unsqueeze(-1))

    return output
