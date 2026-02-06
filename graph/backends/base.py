from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ptx_ops.utils.ir import Graph


BackendEmitter = Callable[[Graph, Path], None]


@dataclass(frozen=True)
class BackendSpec:
    name: str
    implemented: bool
    description: str
