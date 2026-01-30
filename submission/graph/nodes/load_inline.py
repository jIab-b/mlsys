from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..core import Node


@dataclass
class LoadInlineNode(Node):
    def __init__(
        self,
        name: str,
        cuda_src_var: str = "CUDA_SRC",
        cpp_sources: str = "",
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_ldflags: Optional[List[str]] = None,
        verbose: bool = False,
        is_python_module: bool = False,
        no_implicit_headers: bool = True,
        sections: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            kind="LoadInline",
            args={
                "name": name,
                "cuda_src_var": cuda_src_var,
                "cpp_sources": cpp_sources,
                "extra_cuda_cflags": extra_cuda_cflags or [],
                "extra_ldflags": extra_ldflags or [],
                "verbose": verbose,
                "is_python_module": is_python_module,
                "no_implicit_headers": no_implicit_headers,
                "sections": sections or [],
            },
        )
