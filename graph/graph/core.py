from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class MemSpace(Enum):
    GMEM = auto()
    SMEM = auto()
    TMEM = auto()
    RMEM = auto()


class BarrierState(Enum):
    UNINIT = auto()
    INIT = auto()
    PENDING = auto()
    ARRIVED = auto()
    COMPLETE = auto()


class BufferState(Enum):
    EMPTY = auto()
    FILLING = auto()
    FULL = auto()
    CONSUMING = auto()


@dataclass
class Tensor:
    name: str
    space: MemSpace
    shape: Tuple[Any, ...]
    dtype: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Barrier:
    name: str
    scope: str  # "cta" or "cluster"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpContract:
    name: str
    issue_scope: str  # "one_thread", "one_warp", "all_warps"
    pre: Dict[str, BarrierState] = field(default_factory=dict)
    post: Dict[str, BarrierState] = field(default_factory=dict)
    buffer_pre: Dict[str, BufferState] = field(default_factory=dict)
    buffer_post: Dict[str, BufferState] = field(default_factory=dict)


@dataclass
class Node:
    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    loc: Optional["SourceLoc"] = None


@dataclass
class SourceLoc:
    filename: str
    line: int
    column: int = 1


class Graph:
    def __init__(self) -> None:
        self.sections: Dict[str, List[Node]] = {"device": [], "host": [], "python": []}
        self.barriers: Dict[str, Barrier] = {}
        self.buffers: Dict[str, Tensor] = {}
        self.tmaps: Dict[str, Dict[str, Any]] = {}
        self.default_tmem: Optional[str] = None

    def add_barrier(self, name: str, scope: str = "cta") -> None:
        if name in self.barriers:
            existing = self.barriers[name]
            if existing.scope != scope:
                raise ValueError(f"Barrier '{name}' scope mismatch: {existing.scope} vs {scope}")
            return
        self.barriers[name] = Barrier(name=name, scope=scope)

    def add_buffer(self, name: str, space: MemSpace, shape: Tuple[Any, ...], dtype: str) -> None:
        if name in self.buffers:
            existing = self.buffers[name]
            if existing.space != space or existing.dtype != dtype:
                raise ValueError(f"Buffer '{name}' mismatch: {existing.space}/{existing.dtype} vs {space}/{dtype}")
            return
        self.buffers[name] = Tensor(name=name, space=space, shape=shape, dtype=dtype)
        if space == MemSpace.TMEM and self.default_tmem is None:
            self.default_tmem = name

    def add_tmap(self, name: str, meta: Dict[str, Any]) -> None:
        if name in self.tmaps:
            existing = self.tmaps[name]
            if existing.get("rank") != meta.get("rank"):
                raise ValueError(f"TensorMap '{name}' rank mismatch: {existing.get('rank')} vs {meta.get('rank')}")
            existing.update(meta)
            return
        self.tmaps[name] = dict(meta)

    def set_default_tmem(self, name: str) -> None:
        self.default_tmem = name

    def add_node(self, section: str, kind: str, **kwargs: Any) -> Node:
        node = Node(kind=kind, args=kwargs)
        self.sections[section].append(node)
        return node

    def add_raw(self, section: str, code: str, meta: Optional[Dict[str, str]] = None) -> Node:
        from .nodes.raw import RawNode

        node = RawNode(code=code, meta=meta or {})
        self.sections[section].append(node)
        return node

    def add_event(self, section: str, op: str) -> Node:
        from .nodes.event import EventNode

        node = EventNode(op=op)
        self.sections[section].append(node)
        return node

    def add_load_inline(self, section: str, **kwargs: Any) -> Node:
        from .nodes.load_inline import LoadInlineNode

        node = LoadInlineNode(**kwargs)
        self.sections[section].append(node)
        return node

    def block(self, *nodes: Node) -> Node:
        return Node(kind="Block", children=list(nodes))

    def if_(self, cond: str, then_nodes: List[Node], else_nodes: Optional[List[Node]] = None) -> Node:
        node = Node(kind="If", args={"cond": cond})
        node.children.append(Node(kind="Then", children=then_nodes))
        if else_nodes is not None:
            node.children.append(Node(kind="Else", children=else_nodes))
        return node

    def for_(self, var: str, iters: int, body: List[Node]) -> Node:
        return Node(kind="For", args={"var": var, "iters": str(iters)}, children=body)

    def raw(self, code: str, meta: Optional[Dict[str, str]] = None) -> Node:
        return Node(kind="Raw", args={"code": code}, meta=meta or {})
