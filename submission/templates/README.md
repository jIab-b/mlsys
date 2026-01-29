# Kernel Templates

| File | Pattern | When to Use |
|------|---------|-------------|
| `simple_gemm.cuh` | Single-stage TMA+MMA | Baseline |
| `pipelined_gemm.cuh` | Multi-stage pipeline | Memory-bound |
| `warp_specialized_gemm.cuh` | Producer/consumer warps | Compute-bound |
| `persistent_gemm.cuh` | CTAs loop over tiles | Many tiles |
| `split_k_gemm.cuh` | K-parallelism | Small M×N, large K |
| `collector_gemm.cuh` | A-matrix reuse | Tall-skinny, batch |

All use ptx_lib functions only. Tune constants at top of each file.
