from __future__ import annotations

from typing import Any, Optional, Type

from .mbarrier import select_mbarrier_op  # noqa: F401
from .ptx_common import select_ptx_common_op  # noqa: F401
from .tcgen05_cp import select_tcgen05_cp  # noqa: F401
from .tcgen05_mma import select_tcgen05_mma  # noqa: F401
from .tcgen05_st import select_tcgen05_ldst  # noqa: F401
from .tcgen05_sync import select_tcgen05_sync  # noqa: F401
from .tma import select_tma_op  # noqa: F401


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
