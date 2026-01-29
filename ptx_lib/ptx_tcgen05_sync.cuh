#pragma once

#include "ptx_common.cuh"

// tcgen05 commit, wait, fence wrappers (CTA group 1 only for now)

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tcgen05_commit(int mbar_addr) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
              :: "r"(mbar_addr) : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tcgen05_commit_mcast(int mbar_addr, uint16_t cta_mask) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

PTX_DEVICE inline void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

PTX_DEVICE inline void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

PTX_DEVICE inline void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

PTX_DEVICE inline void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

