#pragma once

#include "ptx_common.cuh"

// tcgen05.mma wrappers (CTA group 1 only for now)

struct COLLECTOR_USAGE {
  static constexpr char NONE[]      = "";
  static constexpr char A_FILL[]    = ".collector::a::fill";
  static constexpr char A_USE[]     = ".collector::a::use";
  static constexpr char A_LASTUSE[] = ".collector::a::lastuse";
  static constexpr char A_DISCARD[] = ".collector::a::discard";
};

// Block-scaled NVFP4 MMA (smem A/B descriptors, tmem scales)
template <int CTA_GROUP = 1, const char *collector_usage = COLLECTOR_USAGE::NONE>
PTX_DEVICE inline void tcgen05_mma_mxf4nvf4_block16(
  int d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16%7 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d),
       "C"(collector_usage)
  );
}

// F16/BF16 MMA, SS (A/B in smem desc), C in tmem
PTX_DEVICE inline void tcgen05_mma_f16_ss(
  uint32_t tmem_c,
  uint64_t desc_a,
  uint64_t desc_b,
  uint32_t idesc,
  int accumulate
) {
  PTX_ELECT_ONE();
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
    "}"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate),
       "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
  );
}

// F16/BF16 MMA, TS (A in tmem, B in smem desc), C in tmem
PTX_DEVICE inline void tcgen05_mma_f16_ts(
  uint32_t tmem_c,
  uint32_t tmem_a,
  uint64_t desc_b,
  uint32_t idesc,
  int accumulate
) {
  PTX_ELECT_ONE();
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t"
    "}"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accumulate),
       "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
  );
}

// F16/BF16 MMA, WS (warp-specialized), C in tmem
PTX_DEVICE inline void tcgen05_mma_ws_f16_ts(
  uint32_t tmem_c,
  uint32_t tmem_a,
  uint64_t desc_b,
  uint32_t idesc,
  int accumulate
) {
  PTX_ELECT_ONE();
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n\t"
    "}"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accumulate)
  );
}

