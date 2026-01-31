from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core import Node, SourceLoc


@dataclass
class OpNode(Node):
    def __init__(self, op: str, args: Optional[Dict[str, Any]] = None, loc: Optional[SourceLoc] = None) -> None:
        super().__init__(kind="Op", args={"op": op, "op_args": args or {}}, loc=loc)
