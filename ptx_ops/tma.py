from __future__ import annotations

from typing import Dict, Optional, Type

from graph.nodes.op import OpNode


class TmaOp(OpNode):
    """Base node for TMA / cp.async bulk ops."""


class TmaGmem2Smem(TmaOp):
    pass


class Tma1dGmem2Smem(TmaOp):
    pass


class Tma2dGmem2Smem(TmaOp):
    pass


class Tma3dGmem2Smem(TmaOp):
    pass


class Tma1dGmem2SmemMcast(TmaOp):
    pass


class Tma2dGmem2SmemMcast(TmaOp):
    pass


class Tma3dGmem2SmemMcast(TmaOp):
    pass


class Tma1dSmem2Gmem(TmaOp):
    pass


class Tma2dSmem2Gmem(TmaOp):
    pass


class Tma3dSmem2Gmem(TmaOp):
    pass


class TmaStoreOut(TmaOp):
    pass


TMA_OP_REGISTRY: Dict[str, Type[TmaOp]] = {
    "tma_gmem2smem": TmaGmem2Smem,
    "tma_1d_gmem2smem": Tma1dGmem2Smem,
    "tma_2d_gmem2smem": Tma2dGmem2Smem,
    "tma_3d_gmem2smem": Tma3dGmem2Smem,
    "tma_1d_gmem2smem_mcast": Tma1dGmem2SmemMcast,
    "tma_2d_gmem2smem_mcast": Tma2dGmem2SmemMcast,
    "tma_3d_gmem2smem_mcast": Tma3dGmem2SmemMcast,
    "tma_1d_smem2gmem": Tma1dSmem2Gmem,
    "tma_2d_smem2gmem": Tma2dSmem2Gmem,
    "tma_3d_smem2gmem": Tma3dSmem2Gmem,
    "tma_store_out": TmaStoreOut,
}


def select_tma_op(op_name: str, op_args) -> Optional[Type[TmaOp]]:
    if op_name in TMA_OP_REGISTRY:
        return TMA_OP_REGISTRY[op_name]
    if op_name.startswith("tma_"):
        return TmaOp
    return None
