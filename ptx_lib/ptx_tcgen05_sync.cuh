#pragma once

#include "ptx_common.cuh"

// tcgen05 commit, wait, fence wrappers (CTA group 1 only for now)

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_commit(int mbar_addr) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
              :: "r"(mbar_addr) : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_commit_mcast(int mbar_addr, uint16_t cta_mask) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

// tcgen05 tmem allocation / deallocation (CTA group 1 only for now)
template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_alloc(int smem_addr, int num_cols) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_addr), "r"(num_cols));
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tcgen05_dealloc(int tmem_addr, int num_cols) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
              :: "r"(tmem_addr), "r"(num_cols));
}

PTX_DEVICE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

PTX_DEVICE void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

PTX_DEVICE void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

PTX_DEVICE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
