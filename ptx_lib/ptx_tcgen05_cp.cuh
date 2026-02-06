#pragma once

#include "ptx_common.cuh"

// tcgen05.cp wrappers (CTA group 1/2)

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_cp_32x128b_warpx4(int taddr, uint64_t s_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_cp_128x128b(int taddr, uint64_t s_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(taddr), "l"(s_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x128b [%0], %1;" :: "r"(taddr), "l"(s_desc));
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_cp_128x256b(int taddr, uint64_t s_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(taddr), "l"(s_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x256b [%0], %1;" :: "r"(taddr), "l"(s_desc));
  }
}

// Alias used by gemm1 (NVFP4 block scaling)
template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  tcgen05_cp_32x128b_warpx4<CTA_GROUP>(taddr, s_desc);
}
