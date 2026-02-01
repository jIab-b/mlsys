from __future__ import annotations

from typing import Dict, Optional, Type

from graph.nodes.op import OpNode


class MbarrierOp(OpNode):
    """Base node for mbarrier / cluster barrier ops."""


class MbarrierInit(MbarrierOp):
    pass


class MbarrierArriveExpectTx(MbarrierOp):
    pass


class MbarrierArriveExpectTxCta(MbarrierOp):
    pass


class MbarrierWait(MbarrierOp):
    pass


class MbarrierWaitRelaxed(MbarrierOp):
    pass


class MbarrierWaitTicks(MbarrierOp):
    pass


class MbarrierFenceInitRelease(MbarrierOp):
    pass


class BarrierClusterArriveRelaxedAligned(MbarrierOp):
    pass


class BarrierClusterWaitAcquireAligned(MbarrierOp):
    pass


MBARRIER_OP_REGISTRY: Dict[str, Type[MbarrierOp]] = {
    "mbarrier_init": MbarrierInit,
    "mbarrier_arrive_expect_tx": MbarrierArriveExpectTx,
    "mbarrier_arrive_expect_tx_cta": MbarrierArriveExpectTxCta,
    "mbarrier_wait": MbarrierWait,
    "mbarrier_wait_relaxed": MbarrierWaitRelaxed,
    "mbarrier_wait_ticks": MbarrierWaitTicks,
    "mbarrier_fence_init_release": MbarrierFenceInitRelease,
    "barrier_cluster_arrive_relaxed_aligned": BarrierClusterArriveRelaxedAligned,
    "barrier_cluster_wait_acquire_aligned": BarrierClusterWaitAcquireAligned,
}


def select_mbarrier_op(op_name: str, op_args) -> Optional[Type[MbarrierOp]]:
    if op_name in MBARRIER_OP_REGISTRY:
        return MBARRIER_OP_REGISTRY[op_name]
    if op_name.startswith("mbarrier_") or op_name.startswith("barrier_cluster_"):
        return MbarrierOp
    return None
