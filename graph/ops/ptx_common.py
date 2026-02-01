from __future__ import annotations

from typing import Dict, Optional, Type

from graph.nodes.op import OpNode


class PtxCommonOp(OpNode):
    """Base node for ptx_common helpers."""


class PtxLaneId(PtxCommonOp):
    pass


class PtxActiveMask(PtxCommonOp):
    pass


class PtxElectOneSync(PtxCommonOp):
    pass


class PtxElectSync(PtxCommonOp):
    pass


class PtxBarSync(PtxCommonOp):
    pass


PTX_COMMON_REGISTRY: Dict[str, Type[PtxCommonOp]] = {
    "ptx_laneid": PtxLaneId,
    "ptx_activemask": PtxActiveMask,
    "ptx_elect_one_sync": PtxElectOneSync,
    "ptx_elect_sync": PtxElectSync,
    "ptx_bar_sync": PtxBarSync,
}


def select_ptx_common_op(op_name: str, op_args) -> Optional[Type[PtxCommonOp]]:
    if op_name in PTX_COMMON_REGISTRY:
        return PTX_COMMON_REGISTRY[op_name]
    if op_name.startswith("ptx_"):
        return PtxCommonOp
    return None
