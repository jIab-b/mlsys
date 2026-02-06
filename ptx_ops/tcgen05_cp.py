from __future__ import annotations

from typing import Dict, Optional, Type

from graph.nodes.op import OpNode


class Tcgen05CpOp(OpNode):
    """Base node for tcgen05.cp ops."""


class Tcgen05Cp32x128bWarpx4(Tcgen05CpOp):
    pass


class Tcgen05Cp128x128b(Tcgen05CpOp):
    pass


class Tcgen05Cp128x256b(Tcgen05CpOp):
    pass


class Tcgen05CpNvfp4(Tcgen05CpOp):
    pass


TCGEN05_CP_REGISTRY: Dict[str, Type[Tcgen05CpOp]] = {
    "tcgen05_cp_32x128b_warpx4": Tcgen05Cp32x128bWarpx4,
    "tcgen05_cp_128x128b": Tcgen05Cp128x128b,
    "tcgen05_cp_128x256b": Tcgen05Cp128x256b,
    "tcgen05_cp_nvfp4": Tcgen05CpNvfp4,
}


def select_tcgen05_cp(op_name: str, op_args) -> Optional[Type[Tcgen05CpOp]]:
    if op_name in TCGEN05_CP_REGISTRY:
        return TCGEN05_CP_REGISTRY[op_name]
    if op_name == "tcgen05_cp":
        shape = op_args.get("shape")
        tile = op_args.get("tile")
        if shape == "32x128b" and tile == "warpx4":
            return Tcgen05Cp32x128bWarpx4
        if shape == "128x128b":
            return Tcgen05Cp128x128b
        if shape == "128x256b":
            return Tcgen05Cp128x256b
        return Tcgen05CpOp
    if op_name.startswith("tcgen05_cp"):
        return Tcgen05CpOp
    return None
