


from dataclasses import dataclass
from typing import Dict, Tuple, Type, Union

import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
os.environ.setdefault("TARGET_SM_ARCH", "sm_100a")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

_SF_VEC_SIZE = 16
_MMA_TILER_MN = (128, 64)
_CLUSTER_SHAPE_MN = (1, 2)
_OCCUPANCY = 1
_TMA_CACHE_EVICT_NORMAL = 0x1000000000000000
_TMA_CACHE_EVICT_FIRST = 0x12F0000000000000
_TMA_CACHE_EVICT_LAST = 0x14F0000000000000
class Sm100BlockScaledPersistentDenseGemmKernel:
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        occupancy: int = 1,
        use_approx_sigmoid: bool = False,
        sigmoid_nr_iters: int = 2,
        tma_cache_policy_a: int = _TMA_CACHE_EVICT_FIRST,
        tma_cache_policy_b: int = _TMA_CACHE_EVICT_FIRST,
        tma_cache_policy_sf: int = _TMA_CACHE_EVICT_FIRST,
        max_ab_stage: int = 0,
        max_c_stage: int = 0,
        enable_mainloop_ilp: bool = False,
        enable_sf_s2t_ilp: bool = False,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = int(occupancy)
        self.enable_mainloop_ilp = bool(enable_mainloop_ilp)
        self.enable_sf_s2t_ilp = bool(enable_sf_s2t_ilp)
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 0
        self.use_approx_sigmoid = bool(use_approx_sigmoid)
        self.sigmoid_nr_iters = int(sigmoid_nr_iters)
        self.tma_cache_policy_a = int(tma_cache_policy_a)
        self.tma_cache_policy_b = int(tma_cache_policy_b)
        self.tma_cache_policy_sf = int(tma_cache_policy_sf)
        self.max_ab_stage = int(max_ab_stage)
        self.max_c_stage = int(max_c_stage)
    def _setup_attributes(self):
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.max_ab_stage,
            self.max_c_stage,
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n
        tmem_cols_per_gemm = self.cta_tile_shape_mnk[1] * 2
        tmem_cols_needed = tmem_cols_per_gemm * 2 - self.num_sfa_tmem_cols
        if tmem_cols_needed <= 0:
            self.num_tmem_alloc_cols = 0
        else:
            tmem_cols = (tmem_cols_needed + 31) & -32
            if tmem_cols < 32:
                tmem_cols = 32
            if tmem_cols > 512:
                raise AssertionError("tmem 列数超出硬件限制")
            self.num_tmem_alloc_cols = tmem_cols
    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        max_active_clusters: cutlass.Constexpr,
    ):
        m, n, k, l = problem_size
        sf_k = k // self.sf_vec_size
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, k, l),
                stride=(k, 1, m * k),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, k, l),
                stride=(k, 1, n * k),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
            cute.make_layout(
                (n, k, l),
                stride=(k, 1, n * k),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr, cute.make_layout((m, n, l), stride=(n, 1, m * n))
        )
        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            cute.make_layout((m, sf_k, l), stride=(sf_k, 1, m * sf_k)),
        )
        sfb1_tensor = cute.make_tensor(
            sfb1_ptr,
            cute.make_layout((n, sf_k, l), stride=(sf_k, 1, n * sf_k)),
        )
        sfb2_tensor = cute.make_tensor(
            sfb2_ptr,
            cute.make_layout((n, sf_k, l), stride=(sf_k, 1, n * sf_k)),
        )
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b1_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b1_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(b2_tensor.element_type != self.b_dtype):
            raise TypeError(f"Type must match: {b2_tensor.element_type} != {self.b_dtype}")
        if cutlass.const_expr(sfb1_tensor.element_type != self.sf_dtype):
            raise TypeError(f"Type must match: {sfb1_tensor.element_type} != {self.sf_dtype}")
        if cutlass.const_expr(sfb2_tensor.element_type != self.sf_dtype):
            raise TypeError(f"Type must match: {sfb2_tensor.element_type} != {self.sf_dtype}")
        self._setup_attributes()
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b1_tensor.shape, self.sf_vec_size
        )
        sfb1_tensor = cute.make_tensor(sfb1_tensor.iterator, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_tensor.iterator, sfb_layout)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb1.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb1.shape[0][1], 4)
            new_shape = (
                (
                    tma_tensor_sfb1.shape[0][0],
                    ((2, 2), y)
                ),
                tma_tensor_sfb1.shape[1],
                tma_tensor_sfb1.shape[2]
            )
            x_times_3 = 3 * x
            new_stride = (
                (
                    tma_tensor_sfb1.stride[0][0],
                    ((x, x), x_times_3)
                ),
                tma_tensor_sfb1.stride[1],
                tma_tensor_sfb1.stride[2]
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb1 = cute.make_tensor(tma_tensor_sfb1.iterator, tma_tensor_sfb_new_layout)
            tma_tensor_sfb2 = cute.make_tensor(tma_tensor_sfb2.iterator, tma_tensor_sfb_new_layout)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + (b_copy_size * 2) + sfa_copy_size + (sfb_copy_size * 2)
        ) * atom_thr_size
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )
        self.buffer_align_bytes = 1024
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=self.occupancy,
        )
        return
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB1_nkl: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB2_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb1: cute.CopyAtom,
        mSFB1_nkl: cute.Tensor,
        tma_atom_sfb2: cute.CopyAtom,
        mSFB2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b1)
            cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb1)
            cpasync.prefetch_descriptor(tma_atom_sfb2)
            cpasync.prefetch_descriptor(tma_atom_c)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        try:
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        except TypeError:
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        try:
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        except TypeError:
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB1 = storage.sB1.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB2 = storage.sB2.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB1 = storage.sSFB1.get_tensor(sfb_smem_layout_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb_smem_layout_staged)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.num_mcast_ctas_a > 1):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfa_full_mcast_mask = a_full_mcast_mask
        if cutlass.const_expr(self.num_mcast_ctas_b > 1):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        if cutlass.const_expr(self.num_mcast_ctas_sfb > 1):
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB1_nkl = cute.local_tile(
            mB1_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gB2_nkl = cute.local_tile(
            mB2_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB1_nkl = cute.local_tile(
            mSFB1_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gSFB2_nkl = cute.local_tile(
            mSFB2_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB1 = thr_mma.partition_B(gB1_nkl)
        tCgB2 = thr_mma.partition_B(gB2_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB1 = thr_mma_sfb.partition_B(gSFB1_nkl)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB2_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_atom_b1,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(tCgB1, 0, 3),
        )
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_atom_b2,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB2, 0, 3),
            cute.group_modes(tCgB2, 0, 3),
        )
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB1, tBgSFB1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb1,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB1, 0, 3),
            cute.group_modes(tCgSFB1, 0, 3),
        )
        tBsSFB2, tBgSFB2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb2,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB2, 0, 3),
            cute.group_modes(tCgSFB2, 0, 3),
        )
        tBsSFB1 = cute.filter_zeros(tBsSFB1)
        tBgSFB1 = cute.filter_zeros(tBgSFB1)
        tBsSFB2 = cute.filter_zeros(tBsSFB2)
        tBgSFB2 = cute.filter_zeros(tBgSFB2)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB1 = tiled_mma.make_fragment_B(sB1)
        tCrB2 = tiled_mma.make_fragment_B(sB2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride = (
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (self.cta_tile_shape_mnk[1] - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1]
                    ) 
                )
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB1_slice = tBgB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tBgB2_slice = tBgB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB1_slice = tBgSFB1[(None, slice_n, None, mma_tile_coord_mnl[2])]
                tBgSFB2_slice = tBgSFB2[(None, slice_n, None, mma_tile_coord_mnl[2])]
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    try:
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_a).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                        )
                    try:
                        cute.copy(
                            tma_atom_b1,
                            tBgB1_slice[(None, ab_producer_state.count)],
                            tBsB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_b).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_b1,
                            tBgB1_slice[(None, ab_producer_state.count)],
                            tBsB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                    try:
                        cute.copy(
                            tma_atom_b2,
                            tBgB2_slice[(None, ab_producer_state.count)],
                            tBsB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_b).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_b2,
                            tBgB2_slice[(None, ab_producer_state.count)],
                            tBsB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                    try:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfa_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_sf).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    try:
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, ab_producer_state.count)],
                            tBsSFB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_sf).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, ab_producer_state.count)],
                            tBsSFB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    try:
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, ab_producer_state.count)],
                            tBsSFB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=cutlass.Int64(cutlass.Int64(self.tma_cache_policy_sf).ir_value()),
                        )
                    except TypeError:
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, ab_producer_state.count)],
                            tBsSFB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem_cols_per_gemm = self.cta_tile_shape_mnk[1] * 2
            acc_tmem_ptr1 = acc_tmem_ptr
            acc_tmem_ptr2 = acc_tmem_ptr + tmem_cols_per_gemm
            tCtAcc1_base = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)
            tCtAcc2_base = cute.make_tensor(acc_tmem_ptr2, tCtAcc_fake.layout)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            sfa_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr1 + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            sfb_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr1 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA1 = cute.make_tensor(sfa_tmem_ptr1, tCtSFA_layout)
            tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)
            sfb_tmem_ptr2 = cute.recast_ptr(
                acc_tmem_ptr2 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout)
            (
                tiled_copy_s2t_sfa1,
                tCsSFA1_compact_s2t,
                tCtSFA1_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA1)
            (
                tiled_copy_s2t_sfb1,
                tCsSFB1_compact_s2t,
                tCtSFB1_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB1)
            (
                tiled_copy_s2t_sfb2,
                tCsSFB2_compact_s2t,
                tCtSFB2_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB2, tCtSFB2)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc1 = tCtAcc1_base[(None, None, None, acc_stage_index)]
                tCtAcc2 = tCtAcc2_base[(None, None, None, acc_stage_index)]
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    ab_consumer_state.reset_count()
                    if ab_consumer_state.count < k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                tCtSFB1_mma = tCtSFB1
                tCtSFB2_mma = tCtSFB2
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr1 = cute.recast_ptr(
                        acc_tmem_ptr1 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB1_mma = cute.make_tensor(shifted_ptr1, tCtSFB_layout)
                    shifted_ptr2 = cute.recast_ptr(
                        acc_tmem_ptr2 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB2_mma = cute.make_tensor(shifted_ptr2, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr1 = cute.recast_ptr(
                        acc_tmem_ptr1 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB1_mma = cute.make_tensor(shifted_ptr1, tCtSFB_layout)
                    shifted_ptr2 = cute.recast_ptr(
                        acc_tmem_ptr2 + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB2_mma = cute.make_tensor(shifted_ptr2, tCtSFB_layout)
                if is_leader_cta:
                    for k_tile in range(k_tile_cnt):
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        tCsSFA1_compact_s2t_staged = tCsSFA1_compact_s2t[s2t_stage_coord]
                        tCsSFB1_compact_s2t_staged = tCsSFB1_compact_s2t[s2t_stage_coord]
                        tCsSFB2_compact_s2t_staged = tCsSFB2_compact_s2t[s2t_stage_coord]
                        num_kblocks = cute.size(tCrA, mode=[2])
                        kblock_coord0 = (None, None, 0, ab_consumer_state.index)
                        sf_kblock_coord0 = (None, None, 0)
                        sf_sfa_kblock_coord0 = (None, None, None, 0)
                        sf_sfb_kblock_coord0 = (None, None, None, 0)
                        if cutlass.const_expr(self.enable_sf_s2t_ilp):
                            cute.copy(
                                tiled_copy_s2t_sfa1,
                                tCsSFA1_compact_s2t_staged[sf_sfa_kblock_coord0],
                                tCtSFA1_compact_s2t[sf_sfa_kblock_coord0],
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb1,
                                tCsSFB1_compact_s2t_staged[sf_sfa_kblock_coord0],
                                tCtSFB1_compact_s2t[sf_sfa_kblock_coord0],
                            )
                        else:
                            cute.copy(tiled_copy_s2t_sfa1, tCsSFA1_compact_s2t_staged, tCtSFA1_compact_s2t)
                            cute.copy(tiled_copy_s2t_sfb1, tCsSFB1_compact_s2t_staged, tCtSFB1_compact_s2t)
                        if cutlass.const_expr(not self.enable_mainloop_ilp):
                            cute.copy(tiled_copy_s2t_sfb2, tCsSFB2_compact_s2t_staged, tCtSFB2_compact_s2t)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA1[sf_kblock_coord0].iterator)
                        if ab_consumer_state.count == 0:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB1_mma[sf_kblock_coord0].iterator)
                        cute.gemm(tiled_mma, tCtAcc1, tCrA[kblock_coord0], tCrB1[kblock_coord0], tCtAcc1)
                        if cutlass.const_expr(self.enable_mainloop_ilp):
                            cute.copy(
                                tiled_copy_s2t_sfb2,
                                tCsSFB2_compact_s2t_staged[sf_sfb_kblock_coord0],
                                tCtSFB2_compact_s2t[sf_sfb_kblock_coord0],
                            )
                            if cutlass.const_expr(self.enable_sf_s2t_ilp):
                                if num_kblocks > 1:
                                    sf_sfa_kblock_coord1 = (None, None, None, 1)
                                    cute.copy(
                                        tiled_copy_s2t_sfa1,
                                        tCsSFA1_compact_s2t_staged[sf_sfa_kblock_coord1],
                                        tCtSFA1_compact_s2t[sf_sfa_kblock_coord1],
                                    )
                                    cute.copy(
                                        tiled_copy_s2t_sfb1,
                                        tCsSFB1_compact_s2t_staged[sf_sfa_kblock_coord1],
                                        tCtSFB1_compact_s2t[sf_sfa_kblock_coord1],
                                    )
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB2_mma[sf_kblock_coord0].iterator)
                        cute.gemm(tiled_mma, tCtAcc2, tCrA[kblock_coord0], tCrB2[kblock_coord0], tCtAcc2)
                        if ab_consumer_state.count == 0:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        for kblock_tail in cutlass.range(num_kblocks - 1, unroll_full=True):
                            kblock_idx = kblock_tail + 1
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            sf_sfb_kblock_coord = (None, None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA1[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB1_mma[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc1, tCrA[kblock_coord], tCrB1[kblock_coord], tCtAcc1)
                            if cutlass.const_expr(self.enable_mainloop_ilp):
                                cute.copy(
                                    tiled_copy_s2t_sfb2,
                                    tCsSFB2_compact_s2t_staged[sf_sfb_kblock_coord],
                                    tCtSFB2_compact_s2t[sf_sfb_kblock_coord],
                                )
                                if cutlass.const_expr(self.enable_sf_s2t_ilp):
                                    if kblock_idx + 1 < num_kblocks:
                                        sf_sfa_kblock_coord_pf = (None, None, None, kblock_idx + 1)
                                        cute.copy(
                                            tiled_copy_s2t_sfa1,
                                            tCsSFA1_compact_s2t_staged[sf_sfa_kblock_coord_pf],
                                            tCtSFA1_compact_s2t[sf_sfa_kblock_coord_pf],
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb1,
                                            tCsSFB1_compact_s2t_staged[sf_sfa_kblock_coord_pf],
                                            tCtSFB1_compact_s2t[sf_sfa_kblock_coord_pf],
                                        )
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB2_mma[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc2, tCrA[kblock_coord], tCrB2[kblock_coord], tCtAcc2)
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem_cols_per_gemm = self.cta_tile_shape_mnk[1] * 2
            acc_tmem_ptr1 = acc_tmem_ptr
            acc_tmem_ptr2 = acc_tmem_ptr + tmem_cols_per_gemm
            tCtAcc1_base = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)
            tCtAcc2_base = cute.make_tensor(acc_tmem_ptr2, tCtAcc_fake.layout)
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc1_base,
                tTR_rAcc1,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc1_base, tCgC, epi_tile, use_2cta_instrs
            )
            (
                _tiled_copy_t2r_2,
                tTR_tAcc2_base,
                tTR_rAcc2,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc2_base, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc1.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index
                tTR_tAcc1 = tTR_tAcc1_base[(None, None, None, None, None, acc_stage_index)]
                tTR_tAcc2 = tTR_tAcc2_base[(None, None, None, None, None, acc_stage_index)]
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc1 = cute.group_modes(tTR_tAcc1, 3, cute.rank(tTR_tAcc1))
                tTR_tAcc2 = cute.group_modes(tTR_tAcc2, 3, cute.rank(tTR_tAcc2))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                subtile_cnt = cute.size(tTR_tAcc1.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                    tTR_tAcc1_mn = tTR_tAcc1[(None, None, None, real_subtile_idx)]
                    tTR_tAcc2_mn = tTR_tAcc2[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc1_mn, tTR_rAcc1)
                    cute.copy(tiled_copy_t2r, tTR_tAcc2_mn, tTR_rAcc2)
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                    acc1_vec = tiled_copy_r2s.retile(tTR_rAcc1).load()
                    acc2_vec = tiled_copy_r2s.retile(tTR_rAcc2).load()
                    if cutlass.const_expr(self.use_approx_sigmoid):
                        is_pos = acc1_vec >= 0.0
                        ax = cute.where(is_pos, acc1_vec, -acc1_vec)
                        e = cute.math.exp2(-ax * 1.4426950408889634, fastmath=True)
                        ax = 1.0 + e
                        if cutlass.const_expr(self.sigmoid_nr_iters <= 1):
                            inv = 2.1666666666666665 + ax * (-1.5 + 0.3333333333333333 * ax)
                        else:
                            inv = 1.5 - 0.5 * ax
                        inv = inv * (2.0 - ax * inv)
                        if cutlass.const_expr(self.sigmoid_nr_iters > 1):
                            inv = inv * (2.0 - ax * inv)
                        inv = cute.where(is_pos, inv, e * inv)
                    else:
                        inv = 1.0 / (1.0 + cute.exp(-acc1_vec))
                    out = (acc1_vec * inv) * acc2_vec
                    tRS_rC.store(out.to(self.c_dtype))
                    c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epilog_warp_id[0]:
                        try:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                                cache_policy=cutlass.Int64(cutlass.Int64(_TMA_CACHE_EVICT_NORMAL).ir_value()),
                            )
                        except TypeError:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    if cutlass.const_expr(self.num_c_stage <= 1):
                        self.epilog_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            c_pipeline.producer_tail()
    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t
    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc
    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC
    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC
    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        max_ab_stage: int,
        max_c_stage: int,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1
        num_c_stage = 2
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + 2 * cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + 2 * cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        if max_ab_stage > 0:
            num_ab_stage = min(num_ab_stage, max_ab_stage)
        if max_c_stage > 0:
            num_c_stage = min(num_c_stage, max_c_stage)
        return num_acc_stage, num_ab_stage, num_c_stage
    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid
    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False
        if sf_vec_size not in {16, 32}:
            is_valid = False
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False
        return is_valid
    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        is_valid = True
        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid
    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        is_valid = True
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        def is_power_of_2(x: int) -> bool:
            return x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid
    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        is_valid = True
        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0
        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid
    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        can_implement = True
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


class _RankedSm100BlockScaledPersistentDenseGemmKernel(Sm100BlockScaledPersistentDenseGemmKernel):
    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
    ):
        m, n, k, l = problem_size
        sf_k = k // self.sf_vec_size
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, k, l),
                stride=(k, 1, m * k),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, k, l),
                stride=(k, 1, n * k),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
            cute.make_layout(
                (n, k, l),
                stride=(k, 1, n * k),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr, cute.make_layout((m, n, l), stride=(n, 1, m * n))
        )
        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            cute.make_layout((m, sf_k, l), stride=(sf_k, 1, m * sf_k)),
        )
        sfb1_tensor = cute.make_tensor(
            sfb1_ptr,
            cute.make_layout((n, sf_k, l), stride=(sf_k, 1, n * sf_k)),
        )
        sfb2_tensor = cute.make_tensor(
            sfb2_ptr,
            cute.make_layout((n, sf_k, l), stride=(sf_k, 1, n * sf_k)),
        )
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b1_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b1_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(b2_tensor.element_type != self.b_dtype):
            raise TypeError(f"Type must match: {b2_tensor.element_type} != {self.b_dtype}")
        if cutlass.const_expr(sfb1_tensor.element_type != self.sf_dtype):
            raise TypeError(f"Type must match: {sfb1_tensor.element_type} != {self.sf_dtype}")
        if cutlass.const_expr(sfb2_tensor.element_type != self.sf_dtype):
            raise TypeError(f"Type must match: {sfb2_tensor.element_type} != {self.sf_dtype}")
        self._setup_attributes()
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b1_tensor.shape, self.sf_vec_size)
        sfb1_tensor = cute.make_tensor(sfb1_tensor.iterator, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_tensor.iterator, sfb_layout)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb1.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb1.shape[0][1], 4)
            new_shape = (
                (
                    tma_tensor_sfb1.shape[0][0],
                    ((2, 2), y),
                ),
                tma_tensor_sfb1.shape[1],
                tma_tensor_sfb1.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (
                    tma_tensor_sfb1.stride[0][0],
                    ((x, x), x_times_3),
                ),
                tma_tensor_sfb1.stride[1],
                tma_tensor_sfb1.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb1 = cute.make_tensor(tma_tensor_sfb1.iterator, tma_tensor_sfb_new_layout)
            tma_tensor_sfb2 = cute.make_tensor(tma_tensor_sfb2.iterator, tma_tensor_sfb_new_layout)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + (b_copy_size * 2) + sfa_copy_size + (sfb_copy_size * 2)) * atom_thr_size
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )
        self.buffer_align_bytes = 1024
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=self.occupancy,
        )
        return

@dataclass(frozen=True)
class _GemmCfg:
    mma_tiler_mn: Tuple[int, int]
    cluster_shape_mn: Tuple[int, int]
    occupancy: int
    max_active_clusters: int
    assumed_align: int
    out_dtype: Type[cutlass.Numeric]
    use_approx_sigmoid: bool = False
    sigmoid_nr_iters: int = 2
    tma_cache_policy_a: int = _TMA_CACHE_EVICT_FIRST
    tma_cache_policy_b: int = _TMA_CACHE_EVICT_FIRST
    tma_cache_policy_sf: int = _TMA_CACHE_EVICT_FIRST
    max_ab_stage: int = 0
    max_c_stage: int = 0
    enable_mainloop_ilp: bool = False
    enable_sf_s2t_ilp: bool = False


_GEMM_CACHE: Dict[_GemmCfg, object] = {}
_GEMM_RANKED_CACHE: Dict[Tuple[_GemmCfg, Tuple[int, int, int, int]], object] = {}
_CUTE_COMPILE_OPTIONS_SAFE = "--opt-level 3"
_CUTE_COMPILE_OPTIONS_RANKED = "--opt-level 3"

def _rank_resource_debug_enabled() -> bool:
    
    return False


def _compile_gemm(cfg: _GemmCfg):
    compiled = _GEMM_CACHE.get(cfg)
    if compiled is not None:
        return compiled
    cutlass.cuda.initialize_cuda_context()
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        _SF_VEC_SIZE,
        cfg.mma_tiler_mn,
        cfg.cluster_shape_mn,
        occupancy=cfg.occupancy,
        use_approx_sigmoid=cfg.use_approx_sigmoid,
        sigmoid_nr_iters=cfg.sigmoid_nr_iters,
        tma_cache_policy_a=cfg.tma_cache_policy_a,
        tma_cache_policy_b=cfg.tma_cache_policy_b,
        tma_cache_policy_sf=cfg.tma_cache_policy_sf,
        max_ab_stage=cfg.max_ab_stage,
        max_c_stage=cfg.max_c_stage,
        enable_mainloop_ilp=cfg.enable_mainloop_ilp,
        enable_sf_s2t_ilp=cfg.enable_sf_s2t_ilp,
    )
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b1_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b2_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb1_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb2_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    c_ptr = make_ptr(cfg.out_dtype, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    compiled = cute.compile(
        gemm,
        a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr,
        (0, 0, 0, 0),
        cfg.max_active_clusters,
        options=_CUTE_COMPILE_OPTIONS_SAFE,
    )
    _GEMM_CACHE[cfg] = compiled
    return compiled


def _compile_gemm_ranked(cfg: _GemmCfg, problem_size: Tuple[int, int, int, int]):
    key = (cfg, problem_size)
    compiled = _GEMM_RANKED_CACHE.get(key)
    if compiled is not None:
        return compiled
    cutlass.cuda.initialize_cuda_context()
    gemm = _RankedSm100BlockScaledPersistentDenseGemmKernel(
        _SF_VEC_SIZE,
        cfg.mma_tiler_mn,
        cfg.cluster_shape_mn,
        occupancy=cfg.occupancy,
        use_approx_sigmoid=cfg.use_approx_sigmoid,
        sigmoid_nr_iters=cfg.sigmoid_nr_iters,
        tma_cache_policy_a=cfg.tma_cache_policy_a,
        tma_cache_policy_b=cfg.tma_cache_policy_b,
        tma_cache_policy_sf=cfg.tma_cache_policy_sf,
        max_ab_stage=cfg.max_ab_stage,
        max_c_stage=cfg.max_c_stage,
        enable_mainloop_ilp=cfg.enable_mainloop_ilp,
        enable_sf_s2t_ilp=cfg.enable_sf_s2t_ilp,
    )
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b1_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b2_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb1_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb2_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    c_ptr = make_ptr(cfg.out_dtype, 0, cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    compiled = cute.compile(
        gemm,
        a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr,
        problem_size,
        cfg.max_active_clusters,
        options=_CUTE_COMPILE_OPTIONS_RANKED,
    )
    _GEMM_RANKED_CACHE[key] = compiled
    if _rank_resource_debug_enabled():
        try:
            num_ab_stage = int(getattr(gemm, "num_ab_stage", -1))
            num_c_stage = int(getattr(gemm, "num_c_stage", -1))
            num_tmem_alloc_cols = int(getattr(gemm, "num_tmem_alloc_cols", -1))
            num_tma_load_bytes = int(getattr(gemm, "num_tma_load_bytes", -1))
            cta_tile_shape_mnk = getattr(gemm, "cta_tile_shape_mnk", None)
            m, n, _k, l = problem_size
            tm = int(cta_tile_shape_mnk[0]) if cta_tile_shape_mnk is not None else -1
            tn = int(cta_tile_shape_mnk[1]) if cta_tile_shape_mnk is not None else -1
            m_tiles = (int(m) + tm - 1) // tm if tm > 0 else -1
            n_tiles = (int(n) + tn - 1) // tn if tn > 0 else -1
            cm, cn = cfg.cluster_shape_mn
            clusters_m = (m_tiles + int(cm) - 1) // int(cm) if m_tiles >= 0 else -1
            clusters_n = (n_tiles + int(cn) - 1) // int(cn) if n_tiles >= 0 else -1
            clusters = (clusters_m * clusters_n * int(l)) if clusters_m >= 0 and clusters_n >= 0 else -1
            print(
                f"[rank_res] ps={problem_size} mma={cfg.mma_tiler_mn} cs={cfg.cluster_shape_mn} "
                f"tile={cta_tile_shape_mnk} tiles_mn=({m_tiles},{n_tiles}) clusters={clusters} "
                f"max_active_clusters={cfg.max_active_clusters} ab_stage={num_ab_stage} c_stage={num_c_stage} "
                f"tmem_cols={num_tmem_alloc_cols} tma_bytes={num_tma_load_bytes}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
    return compiled


def _run_fused(
    compiled,
    cfg: _GemmCfg,
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    sfa_p: torch.Tensor,
    sfb1_p: torch.Tensor,
    sfb2_p: torch.Tensor,
    out_c: torch.Tensor,
):
    m, k_half, l = a.shape
    n, k_half_b1, l_b1 = b1.shape
    n2, k_half_b2, l_b2 = b2.shape
    if int(n2) != int(n) or int(k_half_b1) != int(k_half) or int(k_half_b2) != int(k_half):
        raise ValueError("B1/B2 维度不一致")
    if int(l_b1) != int(l) or int(l_b2) != int(l):
        raise ValueError("L 维不一致")
    k = int(k_half) * 2
    a_ptr = make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b1_ptr = make_ptr(cutlass.Float4E2M1FN, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b2_ptr = make_ptr(cutlass.Float4E2M1FN, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, sfa_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb1_ptr = make_ptr(cutlass.Float8E4M3FN, sfb1_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb2_ptr = make_ptr(cutlass.Float8E4M3FN, sfb2_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    c_ptr = make_ptr(cfg.out_dtype, out_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    compiled(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (int(m), int(n), int(k), int(l)))


def _run_fused_ranked(
    compiled,
    cfg: _GemmCfg,
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    sfa_p: torch.Tensor,
    sfb1_p: torch.Tensor,
    sfb2_p: torch.Tensor,
    out_c: torch.Tensor,
):
    a_ptr = make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b1_ptr = make_ptr(cutlass.Float4E2M1FN, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    b2_ptr = make_ptr(cutlass.Float4E2M1FN, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, sfa_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb1_ptr = make_ptr(cutlass.Float8E4M3FN, sfb1_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    sfb2_ptr = make_ptr(cutlass.Float8E4M3FN, sfb2_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    c_ptr = make_ptr(cfg.out_dtype, out_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=cfg.assumed_align)
    compiled(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr)


_CFG_SAFE = _GemmCfg(
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=4096,
    assumed_align=128,
    out_dtype=cutlass.Float16,
)

_CFG_RANK_M256_N4096_TN64_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N4096_TN64_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M256_N4096_TN64_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M256_N4096_TN64_CS21_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M256_N4096_TN64_CS21_A256_ILP_SFILP_CB_LAST_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M256_N4096_TN64_CS21_A256_ILP_SIG1_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M256_N4096_TN64_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
)

_CFG_RANK_M256_N4096_TN64_TM128_CS11_A256 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=6,
    max_c_stage=2,
)

_CFG_RANK_M256_N4096_TN64_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M256_N4096_TN64_TM128_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M256_N4096_TN128_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_CFG_RANK_M256_N4096_TN64_CS21_A256_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N4096_TN64_CS21_A256_CB_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N4096_TN64_CS21_A256_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N4096_TN64_CS21_A256_O2 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N4096_TN64_CS22_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 2),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N4096_TN64_CS24_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 4),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M512_N4096_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_ILP_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_CB_LAST_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_CB_LAST_CSF_NORMAL_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS41_A256_ILP_SIG1_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_O2_AB4 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
)

_EXP_RANK_M512_N4096_CS21_A256_O2_AB4_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_O1_AB4_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_O1_AB6_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=6,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_CS21_A256_O2_AB4_ILP_CB_LAST_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_CFG_RANK_M512_N4096_TN64_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M512_N4096_TN64_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N4096_TN64_CS41_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_CFG_RANK_M512_N4096_TN64_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N4096_TN64_TM128_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N4096_TN128_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N4096_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
)

_CFG_RANK_M512_N4096_CS21_A256_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N4096_CS21_A256_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N4096_CS21_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N4096_CS21_CB_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N4096_CS21_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N4096_CS22 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 2),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_CS24 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 4),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N4096_TN192_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 192),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N4096_TN256_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_TN192_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 192),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_TN256_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N4096_CS21_O2 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 2),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_CS24 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 4),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M256_N3072_CS21_A256_C1 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_c_stage=1,
)

_EXP_RANK_M256_N3072_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M256_N3072_CS21_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M256_N3072_CS21_A256_ILP_SIG1_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M256_N3072_CS21_A256_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_EXP_RANK_M256_N3072_CS21_A256_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N3072_CS22_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 2),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_CS24_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 4),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_TN128_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M256_N3072_TN192_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 192),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M256_N3072_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
)

_CFG_RANK_M256_N3072_TN64_TM128_CS11_A256 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=6,
    max_c_stage=2,
)

_CFG_RANK_M256_N3072_TN64_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M256_N3072_TN64_TM128_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M256_N3072_TN128_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_CFG_RANK_M256_N3072_CS21_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N3072_CS21_CB_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N3072_CS21_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M256_N3072_CS21_O2 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_TN192_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 192),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M256_N3072_TN256_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_CS21_A256_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_EXP_RANK_M512_N3072_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_ILP_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_CB_LAST_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_CB_LAST_CSF_NORMAL_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS41_A256_ILP_SIG1_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_O2_AB4 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
)

_EXP_RANK_M512_N3072_CS21_A256_O2_AB4_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_O1_AB4_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_O1_AB6_ILP = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    max_ab_stage=6,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_CS21_A256_O2_AB4_ILP_CB_LAST_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_LAST,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
    max_ab_stage=4,
    max_c_stage=2,
    enable_mainloop_ilp=True,
)

_CFG_RANK_M512_N3072_TN64_CS21_A256 = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_EXP_RANK_M512_N3072_TN64_CS21_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_TN64_CS41_A256_ILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    enable_mainloop_ilp=True,
)

_EXP_RANK_M512_N3072_TN64_CS41_A256_ILP_SIG1_SFILP = _GemmCfg(
    mma_tiler_mn=(256, 64),
    cluster_shape_mn=(4, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    enable_mainloop_ilp=True,
    enable_sf_s2t_ilp=True,
)

_CFG_RANK_M512_N3072_TN64_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N3072_TN64_TM128_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 64),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N3072_TN128_TM128_CS11_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
    max_ab_stage=6,
    max_c_stage=2,
)

_EXP_RANK_M512_N3072_CS21_A256_SIG1 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=256,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=1,
)

_CFG_RANK_M512_N3072_CS21_CA_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_a=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N3072_CS21_CB_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_b=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N3072_CS21_CSF_NORMAL = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
    tma_cache_policy_sf=_TMA_CACHE_EVICT_NORMAL,
)

_CFG_RANK_M512_N3072_CS22 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 2),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_CS24 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 4),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_TN192_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 192),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_TN256_CS21 = _GemmCfg(
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    occupancy=1,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_CFG_RANK_M512_N3072_CS21_O2 = _GemmCfg(
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    occupancy=2,
    max_active_clusters=176,
    assumed_align=128,
    out_dtype=cutlass.Float16,
    use_approx_sigmoid=True,
    sigmoid_nr_iters=2,
)

_RANK_CFG_TABLE: Dict[Tuple[int, int, int, int], _GemmCfg] = {
    (256, 4096, 7168, 1): _EXP_RANK_M256_N4096_TN64_CS21_A256_ILP_SFILP,
    (512, 4096, 7168, 1): _EXP_RANK_M512_N4096_CS41_A256_ILP_SFILP,
    (256, 3072, 4096, 1): _EXP_RANK_M256_N3072_CS21_A256_ILP_SFILP,
    (512, 3072, 7168, 1): _EXP_RANK_M512_N3072_CS41_A256_ILP_SFILP,
}

_EXP_RANK_CFG_TABLE_TM128: Dict[Tuple[int, int, int, int], _GemmCfg] = {
    (256, 4096, 7168, 1): _CFG_RANK_M256_N4096_TN64_TM128_CS11_A256_SIG1,
    (512, 4096, 7168, 1): _EXP_RANK_M512_N4096_TN128_TM128_CS11_A256_SIG1,
    (256, 3072, 4096, 1): _CFG_RANK_M256_N3072_TN64_TM128_CS11_A256_SIG1,
    (512, 3072, 7168, 1): _EXP_RANK_M512_N3072_TN128_TM128_CS11_A256_SIG1,
}

_RANKED_PROBLEM_SIZES = set(_RANK_CFG_TABLE)


def _assert_ptr_aligned(name: str, tensor: torch.Tensor, align: int) -> None:
    if int(align) <= 1:
        return
    ptr = int(tensor.data_ptr())
    if ptr % int(align) != 0:
        raise RuntimeError(
            f"{name} 指针未满足 assumed_align={int(align)}，ptr%align={ptr % int(align)}"
        )


def _assert_cfg_tmem_budget(cfg: "_GemmCfg") -> None:
    tm = int(cfg.mma_tiler_mn[0])
    tn = int(cfg.mma_tiler_mn[1])
    cta_tm = tm // 2 if tm == 256 else tm
    sf_atom_mn = 32
    mma_inst_tile_k = 4
    num_sfa_cols = (cta_tm // sf_atom_mn) * mma_inst_tile_k
    tmem_cols_needed = tn * 4 - num_sfa_cols
    if tmem_cols_needed <= 0:
        tmem_cols = 0
    else:
        tmem_cols = (tmem_cols_needed + 31) & -32
        if tmem_cols < 32:
            tmem_cols = 32
    if tmem_cols > 512:
        raise RuntimeError(
            f"cfg mma_tiler_mn={cfg.mma_tiler_mn} 预计触发 tmem 列数>512："
            f"cta_tm={cta_tm} num_sfa_cols={num_sfa_cols} tmem_cols_needed={tmem_cols_needed} round={tmem_cols}"
        )


def custom_kernel(data):
    try:
        a, b1, b2, _sfa, _sfb1, _sfb2, sfa_p, sfb1_p, sfb2_p, c = data

        if (
            a.dtype != torch.float4_e2m1fn_x2
            or b1.dtype != torch.float4_e2m1fn_x2
            or b2.dtype != torch.float4_e2m1fn_x2
        ):
            raise TypeError("A/B dtype 不符合 NVFP4 约定")
        if (
            sfa_p.dtype != torch.float8_e4m3fn
            or sfb1_p.dtype != torch.float8_e4m3fn
            or sfb2_p.dtype != torch.float8_e4m3fn
        ):
            raise TypeError("scale dtype 不符合 fp8_e4m3fn 约定")
        if c.dtype != torch.float16:
            raise TypeError("输出 dtype 必须为 fp16")

        m, k_half, l = a.shape
        n, k_half_b1, l_b1 = b1.shape
        n2, k_half_b2, l_b2 = b2.shape
        if int(n2) != int(n) or int(k_half_b1) != int(k_half) or int(k_half_b2) != int(k_half):
            raise ValueError("B1/B2 形状不一致")
        if int(l_b1) != int(l) or int(l_b2) != int(l):
            raise ValueError("L 维不一致")

        k = int(k_half) * 2
        problem_size = (int(m), int(n), int(k), int(l))
        if problem_size in _RANKED_PROBLEM_SIZES:
            try:
                cfg = _RANK_CFG_TABLE[problem_size]
                _assert_ptr_aligned("a", a, cfg.assumed_align)
                _assert_ptr_aligned("b1", b1, cfg.assumed_align)
                _assert_ptr_aligned("b2", b2, cfg.assumed_align)
                _assert_ptr_aligned("sfa_p", sfa_p, cfg.assumed_align)
                _assert_ptr_aligned("sfb1_p", sfb1_p, cfg.assumed_align)
                _assert_ptr_aligned("sfb2_p", sfb2_p, cfg.assumed_align)
                _assert_ptr_aligned("c", c, cfg.assumed_align)
                _assert_cfg_tmem_budget(cfg)
                compiled = _compile_gemm_ranked(cfg, problem_size)
            except Exception as exc:
                raise RuntimeError(f"ranked 编译失败: {type(exc).__name__}: {exc}") from None
            try:
                _run_fused_ranked(compiled, cfg, a, b1, b2, sfa_p, sfb1_p, sfb2_p, c)
            except Exception as exc:
                raise RuntimeError(f"ranked 执行失败: {type(exc).__name__}: {exc}") from None
        else:
            cfg = _CFG_SAFE
            compiled = _compile_gemm(cfg)
            _run_fused(compiled, cfg, a, b1, b2, sfa_p, sfb1_p, sfb2_p, c)
        return c
    except Exception as exc:
        raise RuntimeError(f"custom_kernel 失败: {type(exc).__name__}: {exc}") from None


__all__ = ["custom_kernel"]
