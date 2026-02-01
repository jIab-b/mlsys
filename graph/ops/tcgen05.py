from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from graph.nodes.op import OpNode
from graph.core import SourceLoc


class TcgenOp(OpNode):
    family: str = "tcgen05"
    shape: Optional[str] = None
    tile: Optional[str] = None
    num: Optional[int] = None

    def __init__(self, op: str, args: Optional[Dict[str, Any]] = None, loc: Optional[SourceLoc] = None) -> None:
        super().__init__(op=op, args=args, loc=loc)
        if self.family:
            self.meta["tcgen_family"] = self.family
        if self.shape:
            self.meta["tcgen_shape"] = self.shape
        if self.tile:
            self.meta["tcgen_tile"] = self.tile
        if self.num is not None:
            self.meta["tcgen_num"] = str(self.num)


class TcgenCpOp(TcgenOp):
    family = "tcgen05_cp"


class TcgenCp32x128bWarpx4(TcgenCpOp):
    shape = "32x128b"
    tile = "warpx4"


class TcgenCp128x128b(TcgenCpOp):
    shape = "128x128b"


class TcgenCp128x256b(TcgenCpOp):
    shape = "128x256b"


class TcgenLdOp(TcgenOp):
    family = "tcgen05_ld"


class TcgenLd32x32bX32(TcgenLdOp):
    shape = "32x32b"
    num = 32


class TcgenLd32x32bX64(TcgenLdOp):
    shape = "32x32b"
    num = 64


class TcgenLd32x32bX128(TcgenLdOp):
    shape = "32x32b"
    num = 128


class TcgenLd16x256bX4(TcgenLdOp):
    shape = "16x256b"
    num = 4


class TcgenLd16x256bX8(TcgenLdOp):
    shape = "16x256b"
    num = 8


class TcgenLd16x256bX16(TcgenLdOp):
    shape = "16x256b"
    num = 16


class TcgenMmaOp(TcgenOp):
    family = "tcgen05_mma"


class TcgenMmaMxf4Nvf4Block16(TcgenMmaOp):
    shape = "mxf4nvf4.block16"


TcgenKey = Tuple[str, Optional[str], Optional[str], Optional[int]]

TCGEN_OP_REGISTRY: Dict[TcgenKey, Type[TcgenOp]] = {
    ("tcgen05_cp", "32x128b", "warpx4", None): TcgenCp32x128bWarpx4,
    ("tcgen05_cp", "128x128b", None, None): TcgenCp128x128b,
    ("tcgen05_cp", "128x256b", None, None): TcgenCp128x256b,
    ("tcgen05_ld", "32x32b", None, 32): TcgenLd32x32bX32,
    ("tcgen05_ld", "32x32b", None, 64): TcgenLd32x32bX64,
    ("tcgen05_ld", "32x32b", None, 128): TcgenLd32x32bX128,
    ("tcgen05_ld", "16x256b", None, 4): TcgenLd16x256bX4,
    ("tcgen05_ld", "16x256b", None, 8): TcgenLd16x256bX8,
    ("tcgen05_ld", "16x256b", None, 16): TcgenLd16x256bX16,
    ("tcgen05_mma", "mxf4nvf4.block16", None, None): TcgenMmaMxf4Nvf4Block16,
}


def select_tcgen_op(op_name: str, op_args: Dict[str, Any]) -> Optional[Type[TcgenOp]]:
    shape = op_args.get("shape")
    tile = op_args.get("tile")
    num = op_args.get("num")
    key = (op_name, shape, tile, num if isinstance(num, int) else None)
    return TCGEN_OP_REGISTRY.get(key)
