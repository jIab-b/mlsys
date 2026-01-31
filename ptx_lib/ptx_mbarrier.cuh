#pragma once

#include "ptx_common.cuh"

// mbarrier helpers (CTA scope)

// NOTE: Keep gemm1 semantics (no implicit election in wrappers).
PTX_DEVICE void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// CTA-scope arrive expect_tx (gemm1 uses CTA scope)
PTX_DEVICE void mbarrier_arrive_expect_tx_cta(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

PTX_DEVICE void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

PTX_DEVICE void mbarrier_wait(int mbar_addr, int phase) {
  // gemm1 uses a ticked wait loop and exits when P1 is true.
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n\t"
      :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

// Explicit wait loop with ticks (as in gemm1)
PTX_DEVICE void mbarrier_wait_ticks(int mbar_addr, int phase, uint32_t ticks) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n\t"
      :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

PTX_DEVICE void mbarrier_wait_relaxed(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "WAIT: \n\t"
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra WAIT;\n\t"
      "}\n\t"
      :: "r"(mbar_addr), "r"(phase), "r"(0xFFFFFFFF));
}

PTX_DEVICE void mbarrier_fence_init_release() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Cluster barrier helpers (used to synchronize CTAs in a cluster after mbarrier init)
PTX_DEVICE void barrier_cluster_arrive_relaxed_aligned() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

PTX_DEVICE void barrier_cluster_wait_acquire_aligned() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
