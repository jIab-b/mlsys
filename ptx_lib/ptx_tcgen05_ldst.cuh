#pragma once

#include "ptx_common.cuh"

// tcgen05.ld / tcgen05.st wrappers

// see https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";
  static constexpr char _16x128b[] = ".16x128b";
  static constexpr char _16x256b[] = ".16x256b";
};

struct NUM {
  static constexpr char x1[]   = ".x1";
  static constexpr char x2[]   = ".x2";
  static constexpr char x4[]   = ".x4";
  static constexpr char x8[]   = ".x8";
  static constexpr char x16[]  = ".x16";
  static constexpr char x32[]  = ".x32";
  static constexpr char x64[]  = ".x64";
  static constexpr char x128[] = ".x128";
};

template <int NUM_REGS, const char *SHAPE, int NUM>
PTX_DEVICE inline void tcgen05_ld(float *tmp, int row, int col) {
  int addr = (row << 16) | col;

  if constexpr (NUM_REGS == 1)
  asm volatile("tcgen05.ld.sync.aligned%2.x%3.b32 {%0}, [%1];"
              : "=f"(tmp[0]) : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 2)
  asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0, %1}, [%2];"
              : "=f"(tmp[0]), "=f"(tmp[1]) : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 4)
  asm volatile("tcgen05.ld.sync.aligned%5.x%6.b32 "
              "{%0, %1, %2, %3}, [%4];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 8)
  asm volatile("tcgen05.ld.sync.aligned%9.x%10.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7}, [%8];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 32)
  asm volatile("tcgen05.ld.sync.aligned%33.x%34.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
}

// Limited tcgen05.st variants (b32). Use uint32_t registers.
template <int NUM_REGS, const char *SHAPE, int NUM>
PTX_DEVICE inline void tcgen05_st(uint32_t const* tmp, int row, int col) {
  int addr = (row << 16) | col;

  if constexpr (NUM_REGS == 1)
  asm volatile("tcgen05.st.sync.aligned%2.x%3.b32 [%0], {%1};"
              :: "r"(addr), "r"(tmp[0]), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 2)
  asm volatile("tcgen05.st.sync.aligned%3.x%4.b32 [%0], {%1, %2};"
              :: "r"(addr), "r"(tmp[0]), "r"(tmp[1]), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 4)
  asm volatile("tcgen05.st.sync.aligned%5.x%6.b32 [%0], {%1, %2, %3, %4};"
              :: "r"(addr), "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 8)
  asm volatile("tcgen05.st.sync.aligned%9.x%10.b32 [%0], "
              "{%1, %2, %3, %4, %5, %6, %7, %8};"
              :: "r"(addr), "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]), "r"(tmp[4]), "r"(tmp[5]), "r"(tmp[6]), "r"(tmp[7]), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.st.sync.aligned%17.x%18.b32 [%0], "
              "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
              :: "r"(addr),
                 "r"(tmp[ 0]), "r"(tmp[ 1]), "r"(tmp[ 2]), "r"(tmp[ 3]), "r"(tmp[ 4]), "r"(tmp[ 5]), "r"(tmp[ 6]), "r"(tmp[ 7]),
                 "r"(tmp[ 8]), "r"(tmp[ 9]), "r"(tmp[10]), "r"(tmp[11]), "r"(tmp[12]), "r"(tmp[13]), "r"(tmp[14]), "r"(tmp[15]),
                 "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 32)
  asm volatile("tcgen05.st.sync.aligned%33.x%34.b32 [%0], "
              "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
              "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
              :: "r"(addr),
                 "r"(tmp[ 0]), "r"(tmp[ 1]), "r"(tmp[ 2]), "r"(tmp[ 3]), "r"(tmp[ 4]), "r"(tmp[ 5]), "r"(tmp[ 6]), "r"(tmp[ 7]),
                 "r"(tmp[ 8]), "r"(tmp[ 9]), "r"(tmp[10]), "r"(tmp[11]), "r"(tmp[12]), "r"(tmp[13]), "r"(tmp[14]), "r"(tmp[15]),
                 "r"(tmp[16]), "r"(tmp[17]), "r"(tmp[18]), "r"(tmp[19]), "r"(tmp[20]), "r"(tmp[21]), "r"(tmp[22]), "r"(tmp[23]),
                 "r"(tmp[24]), "r"(tmp[25]), "r"(tmp[26]), "r"(tmp[27]), "r"(tmp[28]), "r"(tmp[29]), "r"(tmp[30]), "r"(tmp[31]),
                 "C"(SHAPE), "n"(NUM));
}

// Convenience wrappers
PTX_DEVICE inline void tcgen05_ld_32x32b(float *tmp, int row, int col, int num) {
  if (num == 1)  tcgen05_ld<1,  SHAPE::_32x32b, 1>(tmp, row, col);
  if (num == 2)  tcgen05_ld<2,  SHAPE::_32x32b, 2>(tmp, row, col);
  if (num == 4)  tcgen05_ld<4,  SHAPE::_32x32b, 4>(tmp, row, col);
  if (num == 8)  tcgen05_ld<8,  SHAPE::_32x32b, 8>(tmp, row, col);
  if (num == 16) tcgen05_ld<16, SHAPE::_32x32b, 16>(tmp, row, col);
  if (num == 32) tcgen05_ld<32, SHAPE::_32x32b, 32>(tmp, row, col);
}

PTX_DEVICE inline void tcgen05_ld_16x128b(float *tmp, int row, int col, int num) {
  if (num == 1)  tcgen05_ld<1,  SHAPE::_16x128b, 1>(tmp, row, col);
  if (num == 2)  tcgen05_ld<2,  SHAPE::_16x128b, 2>(tmp, row, col);
  if (num == 4)  tcgen05_ld<4,  SHAPE::_16x128b, 4>(tmp, row, col);
  if (num == 8)  tcgen05_ld<8,  SHAPE::_16x128b, 8>(tmp, row, col);
  if (num == 16) tcgen05_ld<16, SHAPE::_16x128b, 16>(tmp, row, col);
  if (num == 32) tcgen05_ld<32, SHAPE::_16x128b, 32>(tmp, row, col);
}

PTX_DEVICE inline void tcgen05_ld_16x256b(float *tmp, int row, int col, int num) {
  if (num == 1)  tcgen05_ld<1,  SHAPE::_16x256b, 1>(tmp, row, col);
  if (num == 2)  tcgen05_ld<2,  SHAPE::_16x256b, 2>(tmp, row, col);
  if (num == 4)  tcgen05_ld<4,  SHAPE::_16x256b, 4>(tmp, row, col);
  if (num == 8)  tcgen05_ld<8,  SHAPE::_16x256b, 8>(tmp, row, col);
  if (num == 16) tcgen05_ld<16, SHAPE::_16x256b, 16>(tmp, row, col);
  if (num == 32) tcgen05_ld<32, SHAPE::_16x256b, 32>(tmp, row, col);
}

