#pragma once

#include "ptx_common.cuh"

// mbarrier helpers (CTA scope)

PTX_DEVICE inline void mbarrier_init(int mbar_addr, int count) {
  PTX_ELECT_ONE();
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// CTA-scope arrive expect_tx (gemm1 uses CTA scope)
PTX_DEVICE inline void mbarrier_arrive_expect_tx_cta(int mbar_addr, int size) {
  PTX_ELECT_ONE();
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

PTX_DEVICE inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  PTX_ELECT_ONE();
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

PTX_DEVICE inline void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "WAIT: \n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra WAIT;\n\t"
      "}\n\t"
      :: "r"(mbar_addr), "r"(phase), "r"(0xFFFFFFFF));
}

// Explicit wait loop with ticks (as in gemm1)
PTX_DEVICE inline void mbarrier_wait_ticks(int mbar_addr, int phase, uint32_t ticks) {
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

PTX_DEVICE inline void mbarrier_wait_relaxed(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "WAIT: \n\t"
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra WAIT;\n\t"
      "}\n\t"
      :: "r"(mbar_addr), "r"(phase), "r"(0xFFFFFFFF));
}

PTX_DEVICE inline void mbarrier_fence_init_release() {
  PTX_ELECT_ONE();
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
