// experimental host snippet (placeholder)
inline int exp_host_noop(int x) { return x; }


// Host launch code for grouped GEMM

template <
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> grouped_gemm_launch(
  const std::vector<at::Tensor>& A_list,
  const std::vector<at::Tensor>& B_list,
  const std::vector<at::Tensor>& SFA_list,
  const std::vector<at::Tensor>& SFB_list,
  std::vector<at::Tensor> C_list
) {
  static_assert(BLOCK_K % 256 == 0);
  
  const int num_groups = A_list.size();
  
  // Calculate total sizes and offsets
  std::vector<GroupedProblem> problems(num_groups);
  int64_t total_A_bytes = 0, total_B_bytes = 0;
  int64_t total_SFA_bytes = 0, total_SFB_bytes = 0;
  int64_t total_C_elems = 0;
  int total_tiles = 0;
  
  for (int g = 0; g < num_groups; g++) {
    int M = A_list[g].size(0);
    int N = B_list[g].size(0);
    int K = A_list[g].size(1) * 2;
    
    problems[g].M = M;
    problems[g].N = N;
    problems[g].K = K;
    problems[g].tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    problems[g].tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    problems[g].tile_start = total_tiles;
    
    problems[g].A_offset = total_A_bytes;
    problems[g].B_offset = total_B_bytes;
    problems[g].SFA_offset = total_SFA_bytes;
    problems[g].SFB_offset = total_SFB_bytes;
    problems[g].C_offset = total_C_elems;
    
    total_A_bytes += M * (K / 2);
    total_B_bytes += N * (K / 2);
    total_SFA_bytes += SFA_list[g].numel();
    total_SFB_bytes += SFB_list[g].numel();
    total_C_elems += M * N;
    total_tiles += problems[g].tiles_m * problems[g].tiles_n;
  }
  
  // Allocate packed buffers
  auto opts = at::TensorOptions().dtype(at::kByte).device(A_list[0].device());
  auto A_packed = at::empty({total_A_bytes}, opts);
  auto B_packed = at::empty({total_B_bytes}, opts);
  auto SFA_packed = at::empty({total_SFA_bytes}, opts);
  auto SFB_packed = at::empty({total_SFB_bytes}, opts);
  auto C_packed = at::empty({total_C_elems}, at::TensorOptions().dtype(at::kHalf).device(A_list[0].device()));
  
  // Copy data to packed buffers
  for (int g = 0; g < num_groups; g++) {
    auto A_view = A_packed.narrow(0, problems[g].A_offset, A_list[g].numel());
    auto B_view = B_packed.narrow(0, problems[g].B_offset, B_list[g].numel());
    auto SFA_view = SFA_packed.narrow(0, problems[g].SFA_offset, SFA_list[g].numel());
    auto SFB_view = SFB_packed.narrow(0, problems[g].SFB_offset, SFB_list[g].numel());
    
    A_view.copy_(A_list[g].view({-1}));
    B_view.copy_(B_list[g].view({-1}));
    SFA_view.copy_(SFA_list[g].view({-1}));
    SFB_view.copy_(SFB_list[g].view({-1}));
  }
  
  // Copy problems to device
  auto problems_tensor = at::empty({(int64_t)(num_groups * sizeof(GroupedProblem))}, opts);
  cudaMemcpy(problems_tensor.data_ptr(), problems.data(), 
             num_groups * sizeof(GroupedProblem), cudaMemcpyHostToDevice);
  
  auto A_ptr = reinterpret_cast<const char*>(A_packed.data_ptr());
  auto B_ptr = reinterpret_cast<const char*>(B_packed.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char*>(SFA_packed.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char*>(SFB_packed.data_ptr());
  auto C_ptr = reinterpret_cast<half*>(C_packed.data_ptr());
  auto probs_ptr = reinterpret_cast<const GroupedProblem*>(problems_tensor.data_ptr());
  
  int tb_size = BLOCK_M + 2 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;
  
  auto this_kernel = grouped_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, C_N_MAJOR, NUM_STAGES>;
  if (smem_size > 48000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  this_kernel<<<total_tiles, tb_size, smem_size>>>(
    A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, probs_ptr, num_groups, total_tiles
  );
  
  // Copy results back to individual tensors
  for (int g = 0; g < num_groups; g++) {
    auto C_view = C_packed.narrow(0, problems[g].C_offset, problems[g].M * problems[g].N);
    C_list[g].copy_(C_view.view({problems[g].M, problems[g].N}));
  }
  
  return C_list;
}


std::vector<at::Tensor> grouped_gemm(
  const std::vector<at::Tensor>& A_list,
  const std::vector<at::Tensor>& B_list,
  const std::vector<at::Tensor>& SFA_list,
  const std::vector<at::Tensor>& SFB_list,
  std::vector<at::Tensor> C_list
) {
  if (A_list.empty()) return C_list;
  
  const int K = A_list[0].size(1) * 2;
  if ((K % 256) != 0) {
    TORCH_CHECK(false, "K must be divisible by 256");
  }
  
  // Select kernel configuration based on K
  // Use BLOCK_K=256 for all cases, vary other parameters
#ifdef LAUNCH
#undef LAUNCH
#endif
#define LAUNCH(K_, BLOCK_M, BLOCK_N, BLOCK_K, C_N_MAJOR, NUM_STAGES) \
  else if (K == K_) return grouped_gemm_launch<BLOCK_M, BLOCK_N, BLOCK_K, C_N_MAJOR, NUM_STAGES>( \
    A_list, B_list, SFA_list, SFB_list, C_list);

  if (false) {}
  LAUNCH(7168, 128, 64, 256, true, 8)   // Large K benchmark
  LAUNCH(2048, 128, 64, 256, true, 8)   // Medium K benchmark
  LAUNCH(4096, 128, 64, 256, true, 6)   // Medium K benchmark 2
  LAUNCH(1536, 128, 64, 256, true, 6)   // Small K benchmark
  // Fallback for other K values
  LAUNCH( 256, 128, 64, 256, true, 4)
  LAUNCH( 512, 128, 64, 256, true, 4)
  LAUNCH( 768, 128, 64, 256, true, 4)

#undef LAUNCH
  
  // Default fallback
  return grouped_gemm_launch<128, 64, 256, true, 6>(A_list, B_list, SFA_list, SFB_list, C_list);
}

TORCH_LIBRARY(my_grouped_gemm, m) {
  m.def("grouped_gemm(Tensor[] A, Tensor[] B, Tensor[] SFA, Tensor[] SFB, Tensor(a!)[] C) -> Tensor[]");
  m.impl("grouped_gemm", &grouped_gemm);
}
