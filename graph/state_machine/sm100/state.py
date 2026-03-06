from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from graph.ir import BarrierState, BufferState


@dataclass
class ValidationState:
    bar_state: Dict[str, Optional[BarrierState]]
    buf_state: Dict[str, Optional[BufferState]]
    bar_init_count: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_arrivals: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_phase: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_expected_bytes: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_completed_bytes: Dict[str, Optional[int]] = field(default_factory=dict)
    cluster_init_fenced: Optional[bool] = None
    cluster_sync_done: Optional[bool] = None
    cluster_ctas: Optional[int] = None
    pending_ld: Optional[bool] = False
    pending_st: Optional[bool] = False
    pending_tcgen_commit: Optional[bool] = False
    cta_group: Optional[int] = None
    last_alloc_cols: Optional[int] = None


@dataclass
class KernelProtocolState:
    active: bool = False
    name: str = ""
    bar_expected: Dict[str, Optional[int]] = field(default_factory=dict)
    bar_completed: Dict[str, Optional[int]] = field(default_factory=dict)
    saw_group2: bool = False
    saw_group2_marker: bool = False
