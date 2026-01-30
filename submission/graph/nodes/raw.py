from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..core import Node


@dataclass
class RawNode(Node):
    def __init__(self, code: str, meta: Optional[Dict[str, str]] = None) -> None:
        super().__init__(kind="Raw", args={"code": code}, meta=meta or {})
