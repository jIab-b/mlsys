from __future__ import annotations

from dataclasses import dataclass

from ..core import Node


@dataclass
class EventNode(Node):
    def __init__(self, op: str) -> None:
        super().__init__(kind="Event", args={"op": op})
