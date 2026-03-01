"""Op contracts: per-op pre/post conditions for barrier and buffer state.

Pure ISA reference data. The state machine consumes this; ptx_ops owns it.
"""
from __future__ import annotations

from typing import Dict

from graph.ir import BarrierState, BufferState, OpContract

CONTRACTS: Dict[str, OpContract] = {
    # tcgen05 core
    "tcgen05_alloc": OpContract(
        name="tcgen05_alloc", issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.EMPTY}, buffer_post={"tmem": BufferState.FULL},
    ),
    "tcgen05_dealloc": OpContract(
        name="tcgen05_dealloc", issue_scope="one_warp",
        buffer_pre={"tmem": BufferState.FULL}, buffer_post={"tmem": BufferState.EMPTY},
    ),
    "tcgen05_cp": OpContract(name="tcgen05_cp", issue_scope="one_thread", buffer_pre={"tmem": BufferState.FULL}),
    "tcgen05_mma": OpContract(name="tcgen05_mma", issue_scope="one_thread", buffer_pre={"tmem": BufferState.FULL}),
    "tcgen05_ld": OpContract(name="tcgen05_ld", issue_scope="one_warp", buffer_pre={"tmem": BufferState.FULL}),
    "tcgen05_st": OpContract(name="tcgen05_st", issue_scope="one_warp", buffer_pre={"tmem": BufferState.FULL}),
    "tcgen05_commit": OpContract(name="tcgen05_commit", issue_scope="one_thread"),
    "tcgen05_commit_mcast": OpContract(name="tcgen05_commit_mcast", issue_scope="one_thread"),
    "tcgen05_wait": OpContract(name="tcgen05_wait", issue_scope="one_thread"),
    "tcgen05_wait_ld": OpContract(name="tcgen05_wait_ld", issue_scope="one_thread"),
    "tcgen05_wait_st": OpContract(name="tcgen05_wait_st", issue_scope="one_thread"),
    "tcgen05_fence": OpContract(name="tcgen05_fence", issue_scope="one_thread"),
    "tcgen05_fence_before_thread_sync": OpContract(name="tcgen05_fence_before_thread_sync", issue_scope="one_thread"),
    "tcgen05_fence_after_thread_sync": OpContract(name="tcgen05_fence_after_thread_sync", issue_scope="one_thread"),
    # mbarrier
    "mbarrier_init": OpContract(name="mbarrier_init", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx": OpContract(name="mbarrier_arrive_expect_tx", issue_scope="one_thread"),
    "mbarrier_arrive_expect_tx_cta": OpContract(name="mbarrier_arrive_expect_tx_cta", issue_scope="one_thread"),
    "mbarrier_wait": OpContract(name="mbarrier_wait", issue_scope="all_warps"),
    "mbarrier_wait_ticks": OpContract(name="mbarrier_wait_ticks", issue_scope="all_warps"),
    "mbarrier_wait_relaxed": OpContract(name="mbarrier_wait_relaxed", issue_scope="all_warps"),
    "mbarrier_fence_init_release": OpContract(name="mbarrier_fence_init_release", issue_scope="one_thread"),
    # cluster sync
    "barrier_cluster_arrive": OpContract(name="barrier_cluster_arrive", issue_scope="all_warps"),
    "barrier_cluster_wait": OpContract(name="barrier_cluster_wait", issue_scope="all_warps"),
    # tma (collapsed)
    "tma_load": OpContract(name="tma_load", issue_scope="one_thread"),
    "tma_load_mcast": OpContract(name="tma_load_mcast", issue_scope="one_thread"),
    "tma_store": OpContract(name="tma_store", issue_scope="one_thread"),
    # host-side metadata
    "tmap_create": OpContract(name="tmap_create", issue_scope="host"),
    "cute_tmap": OpContract(name="cute_tmap", issue_scope="host"),
    "cta_group_set": OpContract(name="cta_group_set", issue_scope="one_thread"),
}
