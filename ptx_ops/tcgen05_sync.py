from __future__ import annotations

from typing import Dict, Optional, Type

from ptx_ops.utils.ir import OpNode


class Tcgen05SyncOp(OpNode):
    """Base node for tcgen05 sync/alloc/commit ops."""


class Tcgen05Commit(Tcgen05SyncOp):
    pass


class Tcgen05CommitMcast(Tcgen05SyncOp):
    pass


class Tcgen05Alloc(Tcgen05SyncOp):
    pass


class Tcgen05Dealloc(Tcgen05SyncOp):
    pass


class Tcgen05WaitLd(Tcgen05SyncOp):
    pass


class Tcgen05WaitSt(Tcgen05SyncOp):
    pass


class Tcgen05FenceBeforeThreadSync(Tcgen05SyncOp):
    pass


class Tcgen05FenceAfterThreadSync(Tcgen05SyncOp):
    pass


TCGEN05_SYNC_REGISTRY: Dict[str, Type[Tcgen05SyncOp]] = {
    "tcgen05_commit": Tcgen05Commit,
    "tcgen05_commit_mcast": Tcgen05CommitMcast,
    "tcgen05_alloc": Tcgen05Alloc,
    "tcgen05_dealloc": Tcgen05Dealloc,
    "tcgen05_wait_ld": Tcgen05WaitLd,
    "tcgen05_wait_st": Tcgen05WaitSt,
    "tcgen05_fence_before_thread_sync": Tcgen05FenceBeforeThreadSync,
    "tcgen05_fence_after_thread_sync": Tcgen05FenceAfterThreadSync,
}


def select_tcgen05_sync(op_name: str, op_args) -> Optional[Type[Tcgen05SyncOp]]:
    if op_name in TCGEN05_SYNC_REGISTRY:
        return TCGEN05_SYNC_REGISTRY[op_name]
    if op_name.startswith("tcgen05_commit"):
        return Tcgen05Commit
    if op_name.startswith("tcgen05_alloc"):
        return Tcgen05Alloc
    if op_name.startswith("tcgen05_dealloc"):
        return Tcgen05Dealloc
    if op_name.startswith("tcgen05_wait"):
        return Tcgen05SyncOp
    if op_name.startswith("tcgen05_fence"):
        return Tcgen05SyncOp
    return None
