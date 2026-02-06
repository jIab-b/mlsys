from __future__ import annotations

from typing import Dict, Optional, Type

from ptx_ops.utils.ir import OpNode


class Tcgen05MmaOp(OpNode):
    """Base node for tcgen05.mma ops."""


class Tcgen05MmaMxf4Nvf4Block16(Tcgen05MmaOp):
    pass


class Tcgen05MmaF16SS(Tcgen05MmaOp):
    pass


class Tcgen05MmaF16TS(Tcgen05MmaOp):
    pass


class Tcgen05MmaWsF16TS(Tcgen05MmaOp):
    pass


class Tcgen05MmaNvfp4(Tcgen05MmaOp):
    pass


TCGEN05_MMA_REGISTRY: Dict[str, Type[Tcgen05MmaOp]] = {
    "tcgen05_mma_mxf4nvf4_block16": Tcgen05MmaMxf4Nvf4Block16,
    "tcgen05_mma_f16_ss": Tcgen05MmaF16SS,
    "tcgen05_mma_f16_ts": Tcgen05MmaF16TS,
    "tcgen05_mma_ws_f16_ts": Tcgen05MmaWsF16TS,
    "tcgen05_mma_nvfp4": Tcgen05MmaNvfp4,
}


def select_tcgen05_mma(op_name: str, op_args) -> Optional[Type[Tcgen05MmaOp]]:
    if op_name in TCGEN05_MMA_REGISTRY:
        return TCGEN05_MMA_REGISTRY[op_name]
    if op_name == "tcgen05_mma":
        shape = op_args.get("shape")
        if shape == "mxf4nvf4.block16":
            return Tcgen05MmaMxf4Nvf4Block16
        if shape == "f16.ss":
            return Tcgen05MmaF16SS
        if shape == "f16.ts":
            return Tcgen05MmaF16TS
        if shape == "ws.f16.ts":
            return Tcgen05MmaWsF16TS
        return Tcgen05MmaOp
    if op_name.startswith("tcgen05_mma"):
        return Tcgen05MmaOp
    return None
