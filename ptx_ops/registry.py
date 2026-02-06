from __future__ import annotations

from typing import Any, Optional, Type

from .mbarrier import select_mbarrier_op
from .ptx_common import select_ptx_common_op
from .tcgen05_cp import select_tcgen05_cp
from .tcgen05_mma import select_tcgen05_mma
from .tcgen05_st import select_tcgen05_ldst
from .tcgen05_sync import select_tcgen05_sync
from .tma import select_tma_op


def select_op(op_name: str, op_args: dict[str, Any]) -> Optional[Type[Any]]:
    for selector in (
        select_tcgen05_cp,
        select_tcgen05_mma,
        select_tcgen05_ldst,
        select_tcgen05_sync,
        select_tma_op,
        select_mbarrier_op,
        select_ptx_common_op,
    ):
        cls = selector(op_name, op_args)
        if cls is not None:
            return cls
    return None

