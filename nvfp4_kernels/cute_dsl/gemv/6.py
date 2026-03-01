import torch
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import math
import cuda.bindings.driver as cuda
import ctypes

tiler_m = 32
warp_size = 32

ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E4M3FN  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type


# https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemv_blockscaled.h
@dsl_user_op
def blockscaled_multiply_add(
    frg_a_packed, frg_b_packed, frg_sfa_packed, frg_sfb_packed, *, loc=None, ip=None
) -> cute.Float16:
    out = llvm.inline_asm(
        cute.Float16.mlir_type,
        [
            frg_sfa_packed[0].ir_value(loc=loc, ip=ip),
            frg_sfb_packed[0].ir_value(loc=loc, ip=ip),
            frg_a_packed[0].ir_value(loc=loc, ip=ip),
            frg_b_packed[0].ir_value(loc=loc, ip=ip),
            frg_a_packed[1].ir_value(loc=loc, ip=ip),
            frg_b_packed[1].ir_value(loc=loc, ip=ip),
            frg_a_packed[2].ir_value(loc=loc, ip=ip),
            frg_b_packed[2].ir_value(loc=loc, ip=ip),
            frg_a_packed[3].ir_value(loc=loc, ip=ip),
            frg_b_packed[3].ir_value(loc=loc, ip=ip),
        ],
        # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
        "{\n"
        # declare registers for A / B tensors
        ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\n"
        ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\n"
        ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\n"
        ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\n"
        ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\n"
        ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\n"
        ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\n"
        ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\n"
        # declare registers for accumulators
        ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\n"
        ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\n"
        ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\n"
        ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\n"
        # declare registers for scaling factors
        ".reg .f16x2 sfa_f16x2;\n" ".reg .f16x2 sfb_f16x2;\n" ".reg .f16x2 sf_f16x2;\n"
        # declare registers for conversion
        ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\n"
        ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\n"
        ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\n"
        ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\n"
        ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\n"
        ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\n"
        ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\n"
        ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\n"
        ".reg .f16 result_f16, lane0, lane1;\n"
        ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\n"
        # convert scaling factors from fp8 to f16x2
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, $1;\n" "cvt.rn.f16x2.e4m3x2 sfb_f16x2, $2;\n"
        # clear accumulators
        "mov.b32 accum_0_0, 0;\n"
        "mov.b32 accum_0_1, 0;\n"
        "mov.b32 accum_0_2, 0;\n"
        "mov.b32 accum_0_3, 0;\n"
        "mov.b32 accum_1_0, 0;\n"
        "mov.b32 accum_1_1, 0;\n"
        "mov.b32 accum_1_2, 0;\n"
        "mov.b32 accum_1_3, 0;\n"
        "mov.b32 accum_2_0, 0;\n"
        "mov.b32 accum_2_1, 0;\n"
        "mov.b32 accum_2_2, 0;\n"
        "mov.b32 accum_2_3, 0;\n"
        "mov.b32 accum_3_0, 0;\n"
        "mov.b32 accum_3_1, 0;\n"
        "mov.b32 accum_3_2, 0;\n"
        "mov.b32 accum_3_3, 0;\n"
        # multiply, unpacking and permuting scale factors
        "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\n"
        "mov.b32 {lane0, lane1}, sf_f16x2;\n"
        "mov.b32 mul_f16x2_0, {lane0, lane0};\n"
        "mov.b32 mul_f16x2_1, {lane1, lane1};\n"
        # unpacking A and B tensors
        "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, $3;\n"
        "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, $4;\n"
        "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, $5;\n"
        "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, $6;\n"
        "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, $7;\n"
        "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, $8;\n"
        "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, $9;\n"
        "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, $10;\n"
        # convert A and B tensors from fp4 to f16x2
        # A[0 - 7] and B[0 - 7]
        "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\n"
        # A[8 - 15] and B[8 - 15]
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\n"
        # A[16 - 23] and B[16 - 23]
        "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\n"
        # A[24 - 31] and B[24 - 31]
        "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\n"
        # fma for A[0 - 7] and B[0 - 7]
        "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_0_4, accum_0_0;\n"
        "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_0_5, accum_0_1;\n"
        "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_0_6, accum_0_2;\n"
        "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_0_7, accum_0_3;\n"
        # fma for A[8 - 15] and B[8 - 15]
        "fma.rn.f16x2 accum_1_0, cvt_1_0, cvt_1_4, accum_1_0;\n"
        "fma.rn.f16x2 accum_1_1, cvt_1_1, cvt_1_5, accum_1_1;\n"
        "fma.rn.f16x2 accum_1_2, cvt_1_2, cvt_1_6, accum_1_2;\n"
        "fma.rn.f16x2 accum_1_3, cvt_1_3, cvt_1_7, accum_1_3;\n"
        # fma for A[16 - 23] and B[16 - 23]
        "fma.rn.f16x2 accum_2_0, cvt_2_0, cvt_2_4, accum_2_0;\n"
        "fma.rn.f16x2 accum_2_1, cvt_2_1, cvt_2_5, accum_2_1;\n"
        "fma.rn.f16x2 accum_2_2, cvt_2_2, cvt_2_6, accum_2_2;\n"
        "fma.rn.f16x2 accum_2_3, cvt_2_3, cvt_2_7, accum_2_3;\n"
        # fma for A[24 - 31] and B[24 - 31]
        "fma.rn.f16x2 accum_3_0, cvt_3_0, cvt_3_4, accum_3_0;\n"
        "fma.rn.f16x2 accum_3_1, cvt_3_1, cvt_3_5, accum_3_1;\n"
        "fma.rn.f16x2 accum_3_2, cvt_3_2, cvt_3_6, accum_3_2;\n"
        "fma.rn.f16x2 accum_3_3, cvt_3_3, cvt_3_7, accum_3_3;\n"
        # tree reduction for accumulators
        "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\n"
        "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\n"
        "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\n"
        "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\n"
        "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\n"
        "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\n"
        "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\n"
        "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\n"
        "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_2;\n"
        "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_2;\n"
        "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_2;\n"
        "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_2;\n"
        "add.rn.f16x2 accum_0_0, accum_0_0, accum_1_0;\n"
        "add.rn.f16x2 accum_2_0, accum_2_0, accum_3_0;\n"
        # apply scaling factors and final reduction
        "mul.rn.f16x2 accum_0_0, mul_f16x2_0, accum_0_0;\n"
        "mul.rn.f16x2 accum_2_0, mul_f16x2_1, accum_2_0;\n"
        "add.rn.f16x2 accum_0_0, accum_0_0, accum_2_0;\n"
        "mov.b32 {lane0, lane1}, accum_0_0;\n"
        "add.rn.f16 result_f16, lane0, lane1;\n"
        "mov.b16 $0, result_f16;\n"
        "}\n",
        "=h, h, h, r, r, r, r, r, r, r, r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return out


def warp_reduce(val: cute.Numeric) -> cute.Numeric:
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << 0)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << 1)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << 2)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << 3)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << 4)
    return val


@cute.kernel
def kernel_cute(g_a, g_b, g_sfa, g_sfb, g_c):
    bidx, bidy, _ = cute.arch.block_idx()
    b_g_a = g_a[(None, None, None), (bidx, 0, bidy)]
    b_g_b = g_b[(None, None), (0, bidy)]
    b_g_sfa = g_sfa[(None, None, None), (bidx, 0, bidy)]
    b_g_sfb = g_sfb[(None, None), (0, bidy)]
    b_g_c = g_c[(None, None), (bidx, bidy)]

    tidx, tidy, _ = cute.arch.thread_idx()

    b_g_a = b_g_a[(tidy, None, 0)]
    b_g_b = b_g_b[(None, 0)]
    b_g_sfa = b_g_sfa[(tidy, None, 0)]
    b_g_sfb = b_g_sfb[(None, 0)]
    b_g_c = b_g_c[(tidy, None)]

    odd = (cute.size(b_g_a) // 1024) % 2 == 1
    coal_load = 32 if odd else 64
    tiler_n = 32 if odd else 64
    tv_layout = cute.make_layout(
        (warp_size, (coal_load, tiler_n // coal_load)),
        stride=(coal_load, (1, coal_load * warp_size)),
    )
    coal_load_sf = coal_load // 16
    tiler_n_sf = tiler_n // 16
    tv_layout_sf = cute.make_layout(
        (warp_size, (coal_load_sf, tiler_n_sf // coal_load_sf)),
        stride=(coal_load_sf, (1, coal_load_sf * warp_size)),
    )
    copy_atom_fp4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), b_g_a.element_type, num_bits_per_copy=64
    )
    copy_atom_fp8 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        b_g_sfa.element_type,
        num_bits_per_copy=16 if odd else 32,
    )

    tiled_copy = cute.make_tiled_copy(
        copy_atom_fp4, tv_layout, (tiler_n * tiler_m,)
    ).get_slice(tidx)
    tiled_copy_sf = cute.make_tiled_copy(
        copy_atom_fp8, tv_layout_sf, (tiler_n_sf * tiler_m,)
    ).get_slice(tidx)
    thr_g_a = tiled_copy.partition_S(b_g_a)
    thr_g_b = tiled_copy.partition_S(b_g_b)
    thr_g_sfa = tiled_copy_sf.partition_S(b_g_sfa)
    thr_g_sfb = tiled_copy_sf.partition_S(b_g_sfb)

    bound = cute.size(thr_g_a, mode=[1])
    total_stages = min(bound, 3)
    coord = (None, 0)

    frg_a = cute.make_fragment((thr_g_a.shape[0], total_stages), thr_g_a.element_type)
    frg_b = cute.make_fragment((thr_g_b.shape[0], total_stages), thr_g_b.element_type)
    frg_sfa = cute.make_fragment(
        (thr_g_sfa.shape[0], total_stages), thr_g_sfa.element_type
    )
    frg_sfb = cute.make_fragment(
        (thr_g_sfb.shape[0], total_stages), thr_g_sfb.element_type
    )

    # prefetch loop
    for i in cutlass.range_constexpr(total_stages - 1):
        coord = (None, i)
        coord_s = (None, i % total_stages)
        cute.copy(tiled_copy, thr_g_a[coord], frg_a[coord_s])
        cute.copy(tiled_copy, thr_g_b[coord], frg_b[coord_s])
        cute.copy(tiled_copy_sf, thr_g_sfa[coord], frg_sfa[coord_s])
        cute.copy(tiled_copy_sf, thr_g_sfb[coord], frg_sfb[coord_s])

    out = cute.Float32(0)
    for i in cutlass.range_constexpr(bound):
        coord = (None, i)
        coord_s = (None, i % total_stages)
        coord_sn = (None, (i + total_stages - 1) % total_stages)
        coord_n = (None, i + total_stages - 1)
        if i + total_stages - 1 < bound:
            if total_stages == 1:
                cute.copy(tiled_copy, thr_g_a[coord_n], frg_a[coord_sn])
                cute.copy(tiled_copy_sf, thr_g_sfb[coord_n], frg_sfb[coord_sn])
            cute.copy(tiled_copy, thr_g_b[coord_n], frg_b[coord_sn])
            cute.copy(tiled_copy_sf, thr_g_sfa[coord_n], frg_sfa[coord_sn])

        frg_a_packed = cute.flat_divide(
            cute.recast_tensor(frg_a[coord_s], cute.Uint32), (4,)
        )
        frg_b_packed = cute.flat_divide(
            cute.recast_tensor(frg_b[coord_s], cute.Uint32), (4,)
        )
        frg_sfa_packed = cute.flat_divide(
            cute.recast_tensor(frg_sfa[coord_s], cute.Uint16), (1,)
        )
        frg_sfb_packed = cute.flat_divide(
            cute.recast_tensor(frg_sfb[coord_s], cute.Uint16), (1,)
        )
        out += blockscaled_multiply_add(
            frg_a_packed[(None, 0)],
            frg_b_packed[(None, 0)],
            frg_sfa_packed[(None, 0)],
            frg_sfb_packed[(None, 0)],
        )
        if i + total_stages - 1 < bound:
            if total_stages > 1:
                cute.copy(tiled_copy, thr_g_a[coord_n], frg_a[coord_sn])
                cute.copy(tiled_copy_sf, thr_g_sfb[coord_n], frg_sfb[coord_sn])
        if not odd:
            out += blockscaled_multiply_add(
                frg_a_packed[(None, 1)],
                frg_b_packed[(None, 1)],
                frg_sfa_packed[(None, 1)],
                frg_sfb_packed[(None, 1)],
            )
    out = warp_reduce(out)
    lane_idx = cute.arch.lane_idx()
    if lane_idx == 0:
        b_g_c[0] = cute.Float16(out)
    # tiled_copy_sf = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn_lin // 16).get_slice(tidx)


@cute.jit
def solve_cute(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream: cuda.CUstream,
    m: cutlass.Constexpr[int],
    k: cutlass.Constexpr[int],
    l: cutlass.Constexpr[int],
):
    # Create CuTe Tensor via pointer and problem size.
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, k, l),
            stride=(k, 1, m * k),
        ),
    )
    # We use n=128 to create the torch tensor to do fp4 computation via torch._scaled_mm
    # then copy torch tensor to cute tensor for cute customize kernel computation
    # therefore we need to ensure b_tensor has the right stride with this 128 padded size on n.
    n_padded_128 = 128
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (k, l),
            stride=(1, n_padded_128 * k),
        ),
    )
    c = cute.make_tensor(c_ptr, cute.make_layout((m, l), stride=(1, m)))

    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, k // 16, l),
            stride=(k // 16, 1, m * k // 16),
        ),
    )

    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (k // 16, l),
            stride=(1, n_padded_128 * k // 16),
        ),
    )

    assert cute.size(a, mode=[0]) % 1024 == 0
    assert cute.size(a, mode=[1]) % 1024 == 0

    tiler_n = cute.size(a, mode=[1])
    tiler_block = 2 if cute.size(a, mode=[1]) <= 2048 else 1
    g_a = cute.zipped_divide(a, (tiler_block, tiler_n, 1))
    g_b = cute.zipped_divide(b, (tiler_n, 1))
    g_sfa = cute.zipped_divide(sfa, (tiler_block, tiler_n // 16, 1))
    g_sfb = cute.zipped_divide(sfb, (tiler_n // 16, 1))
    g_c = cute.zipped_divide(c, (tiler_block, 1))

    assert tiler_m == warp_size

    kernel_cute(g_a, g_b, g_sfa, g_sfb, g_c).launch(
        grid=[cute.size(a, mode=[0]) // tiler_block, cute.size(a, mode=[2]), 1],
        block=[tiler_m, tiler_block, 1],
        stream=stream,
    )


# Scaling factor vector size
sf_vec_size = 16


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data,
):
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape
    c_out = torch.empty_like(c_ref)

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_out[:, 0, l_idx] = res[:, 0]
    return c_out


def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.

    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [1, k, l] - Input vector in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [1, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, 1, l] - Output vector in torch.float16 data type
    """
    torch.manual_seed(seed)

    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    # Pad b tensor's N dimension to 128 to call torch._scaled_mm for nvfp4 dot product computation
    b_ref = torch.randint(
        0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.empty((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(
            0, 3, ref_shape, dtype=torch.int8, device="cuda"
        )
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(
            0, 3, mma_shape, dtype=torch.int8, device="cuda"
        )
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(
            *mma_permute_order
        )

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device="cuda")
        j_idx = torch.arange(sf_k, device="cuda")
        b_idx = torch.arange(l, device="cuda")

        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[
            mm32, mm4, mm, kk4, kk, b_grid
        ] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref = sfa_ref_cpu.to("cuda")
    sfb_ref = sfb_ref_cpu.to("cuda")

    return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)


a_ptr_ = ctypes.c_void_p(0)
b_ptr_ = ctypes.c_void_p(0)
c_ptr_ = ctypes.c_void_p(0)
sfa_ptr_ = ctypes.c_void_p(0)
sfb_ptr_ = ctypes.c_void_p(0)
exe_args = None
torch_stream = torch.cuda.current_stream()
current_stream = cuda.CUstream(torch_stream.cuda_stream)
compiled_cute = None
compiled_key = ""


def compile_kernel(data):
    global exe_args, a_ptr_, b_ptr_, c_ptr_, sfa_ptr_, sfb_ptr_, compiled_cute, current_stream, compiled_key
    a, b, sfa, sfb, _, _, c = data
    m, k, l = a.shape
    k = k * 2
    key = f"{m}_{k}_{l}"
    # print("Compiling", m, k, l)
    odd = ((k // 1024) % 2) == 1
    a_ptr = cute.runtime.make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=8 if odd else 16
    )
    b_ptr = cute.runtime.make_ptr(
        ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=8 if odd else 16
    )
    c_ptr = cute.runtime.make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfa_ptr = cute.runtime.make_ptr(
        sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    sfb_ptr = cute.runtime.make_ptr(
        sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    args = (a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, current_stream, m, k, l)
    compiled_cute = cute.compile(solve_cute, *args)
    compiled_key = key
    args = (a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, current_stream)
    compiled_cute(*args)

    solve_cute_compiled = compiled_cute

    if hasattr(solve_cute_compiled, "_default_executor"):
        exe_args = solve_cute_compiled._default_executor._get_invoke_packed_args(
            solve_cute_compiled._default_executor.generate_execution_args(*args)[0]
        )
    else:
        exe_args = solve_cute_compiled.get_invoke_packed_args(
            solve_cute_compiled.generate_execution_args(
                args, {}, solve_cute_compiled.args_spec
            )[0]
        )
    exe_args[0] = ctypes.addressof(a_ptr_)
    exe_args[1] = ctypes.addressof(b_ptr_)
    exe_args[2] = ctypes.addressof(sfa_ptr_)
    exe_args[3] = ctypes.addressof(sfb_ptr_)
    exe_args[4] = ctypes.addressof(c_ptr_)
    return c


def custom_kernel(data):
    global exe_args, a_ptr_, b_ptr_, c_ptr_, sfa_ptr_, sfb_ptr_, compiled_cute, current_stream, compiled_key
    a, b, sfa, sfb, _, _, c = data
    m, k, l = a.shape
    k = k * 2
    key = f"{m}_{k}_{l}"
    if m <= 1024 or k <= 1024 or m % 1024 != 0 or k % 1024 != 0:
        print("Using reference kernel for m:", m, "k:", k, "l:", l)
        return ref_kernel(data)
    if compiled_key != key:
        compile_kernel(data)

    a_ptr_.value = a.data_ptr()
    b_ptr_.value = b.data_ptr()
    sfa_ptr_.value = sfa.data_ptr()
    sfb_ptr_.value = sfb.data_ptr()
    c_ptr_.value = c.data_ptr()
    compiled_cute.capi_func(exe_args)
    return c


import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional

import torch.cuda
from cutlass.cute.nvgpu.common import OpError
from torch.cuda.nvtx import range as nvtx_range


TestSpec = dict


def clear_l2_cache():
    dummy = torch.randn((1024, 1024, 1024), device="cuda")
    del dummy


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def check_implementation(check_copy, output):
    ref_output = ref_kernel(check_copy)
    passed = torch.allclose(ref_output, output)
    if passed:
        return True, ""
    else:
        return False, f"Check failed {ref_output} {output} {ref_output - output}"


def _run_single_benchmark(
    test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """

    durations = []
    # generate input data once
    data = generate_input(**test.args)
    check_copy = _clone_data(data)

    #  first, one obligatory correctness check
    try:
        output = custom_kernel(_clone_data(data))
    except OpError as E:
        return f"Encountered {E}"
    good, message = check_implementation(check_copy, output)
    if not good:
        return message

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            # ensure we use a different seed for every benchmark
            if "seed" in test.args:
                test.args["seed"] += 13

            data = generate_input(**test.args)
            check_copy = _clone_data(data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        clear_l2_cache()

        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) * 1e6  # Convert ms to ns

        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

        del output
        durations.append(duration)

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:
            stats = calculate_stats(durations)

            if (
                stats.err / stats.mean < 0.0001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return calculate_stats(durations)


def run_single_benchmark(
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
):
    return _run_single_benchmark(test, recheck, max_repeats, max_time_ns)


def score(spec, sc):
    score_base = {"1": 21.7, "2": 34.7, "3": 15.4}
    return (score_base[spec] / sc * 1000 - 1) * 21


def run_benchmarking(tests: list[TestCase]):
    run_single_benchmark(tests[0], False, 500, 10e7)

    passed = True
    for idx, test in enumerate(tests):
        result = run_single_benchmark(test, False, 500, 10e9)
        if isinstance(result, Stats):
            print(f"Time {test.spec}: {score(test.spec, result.mean)}")
        else:
            passed = False
            print(f"Test failed: {result}")

    if passed:
        return 0
    else:
        return 112


def solve(*args):
    run_benchmarking(
        [
            TestCase(args={"m": 7168, "k": 16384, "l": 1, "seed": 1111}, spec="1"),
            TestCase(args={"m": 4096, "k": 7168, "l": 8, "seed": 1111}, spec="2"),
            TestCase(args={"m": 7168, "k": 2048, "l": 4, "seed": 1111}, spec="3"),
        ]
    )
