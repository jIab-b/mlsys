from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from ptx_ops.utils.ir import OpNode


class Tcgen05LdStOp(OpNode):
    """Base node for tcgen05.ld/st ops."""


class Tcgen05Ld(Tcgen05LdStOp):
    pass


class Tcgen05Ld16Regs(Tcgen05LdStOp):
    pass


class Tcgen05Ld32Regs(Tcgen05LdStOp):
    pass


class Tcgen05Ld64Regs(Tcgen05LdStOp):
    pass


class Tcgen05Ld128Regs(Tcgen05LdStOp):
    pass


class Tcgen05Ld32x32b(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x128b(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x256b(Tcgen05LdStOp):
    pass


class Tcgen05Ld32x32bX32(Tcgen05LdStOp):
    pass


class Tcgen05Ld32x32bX64(Tcgen05LdStOp):
    pass


class Tcgen05Ld32x32bX128(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x128bX8(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x128bX16(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x128bX32(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x256bX4(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x256bX8(Tcgen05LdStOp):
    pass


class Tcgen05Ld16x256bX16(Tcgen05LdStOp):
    pass


class Tcgen05St(Tcgen05LdStOp):
    pass


TCGEN05_LDST_REGISTRY: Dict[str, Type[Tcgen05LdStOp]] = {
    "tcgen05_ld_16regs": Tcgen05Ld16Regs,
    "tcgen05_ld_32regs": Tcgen05Ld32Regs,
    "tcgen05_ld_64regs": Tcgen05Ld64Regs,
    "tcgen05_ld_128regs": Tcgen05Ld128Regs,
    "tcgen05_ld_32x32b": Tcgen05Ld32x32b,
    "tcgen05_ld_16x128b": Tcgen05Ld16x128b,
    "tcgen05_ld_16x256b": Tcgen05Ld16x256b,
    "tcgen05_ld_32x32bx32": Tcgen05Ld32x32bX32,
    "tcgen05_ld_32x32bx64": Tcgen05Ld32x32bX64,
    "tcgen05_ld_32x32bx128": Tcgen05Ld32x32bX128,
    "tcgen05_ld_16x128bx8": Tcgen05Ld16x128bX8,
    "tcgen05_ld_16x128bx16": Tcgen05Ld16x128bX16,
    "tcgen05_ld_16x128bx32": Tcgen05Ld16x128bX32,
    "tcgen05_ld_16x256bx4": Tcgen05Ld16x256bX4,
    "tcgen05_ld_16x256bx8": Tcgen05Ld16x256bX8,
    "tcgen05_ld_16x256bx16": Tcgen05Ld16x256bX16,
    "tcgen05_st": Tcgen05St,
}

_TCGEN05_LD_SHAPE_NUM: Dict[Tuple[str, int], Type[Tcgen05LdStOp]] = {
    ("32x32b", 32): Tcgen05Ld32x32bX32,
    ("32x32b", 64): Tcgen05Ld32x32bX64,
    ("32x32b", 128): Tcgen05Ld32x32bX128,
    ("16x128b", 8): Tcgen05Ld16x128bX8,
    ("16x128b", 16): Tcgen05Ld16x128bX16,
    ("16x128b", 32): Tcgen05Ld16x128bX32,
    ("16x256b", 4): Tcgen05Ld16x256bX4,
    ("16x256b", 8): Tcgen05Ld16x256bX8,
    ("16x256b", 16): Tcgen05Ld16x256bX16,
}


def select_tcgen05_ldst(op_name: str, op_args) -> Optional[Type[Tcgen05LdStOp]]:
    if op_name in TCGEN05_LDST_REGISTRY:
        return TCGEN05_LDST_REGISTRY[op_name]
    if op_name == "tcgen05_ld":
        shape = op_args.get("shape")
        num = op_args.get("num")
        if isinstance(shape, str) and isinstance(num, int):
            cls = _TCGEN05_LD_SHAPE_NUM.get((shape, num))
            if cls is not None:
                return cls
        return Tcgen05Ld
    if op_name == "tcgen05_st":
        return Tcgen05St
    if op_name.startswith("tcgen05_ld") or op_name.startswith("tcgen05_st"):
        return Tcgen05LdStOp
    return None
