"""Registry of kernel graphs.

Each entry maps a kernel name to its .graph file and metadata.
LLM can use this to find relevant kernels by name/tags.
"""
from pathlib import Path

KERNEL_GRAPHS_DIR = Path(__file__).parent

# Kernel registry: name -> metadata
# Add new kernels here as they are created
KERNELS = {
    "gemm1": {
        "graph": "gemm1.graph",
        "description": "Typed graph IR for NVFP4 block-scaled GEMM with explicit TMA/tcgen05 protocol nodes",
        "tags": ["gemm", "nvfp4", "tma", "tcgen05", "warpspecialized", "single_cta", "typed_graph"],
        "kernels": ["kernel_v4", "kernel_v3b"],  # kernel functions defined in this graph
    },
    "grouped_gemm": {
        "graph": "grouped_gemm.graph",
        "description": "Typed graph IR for grouped NVFP4 GEMM with explicit CTA-group-2 and persistent loop markers",
        "tags": ["gemm", "nvfp4", "tma", "tcgen05", "warpspecialized", "grouped", "typed_graph", "cta_group2", "persistent"],
        "kernels": ["grouped_gemm_kernel"],
    },
}


def list_kernels(tags: list[str] | None = None) -> list[str]:
    """List kernel names, optionally filtered by tags."""
    if tags is None:
        return list(KERNELS.keys())
    return [
        name for name, meta in KERNELS.items()
        if all(t in meta.get("tags", []) for t in tags)
    ]


def get_graph_path(name: str) -> Path:
    """Get the .graph file path for a kernel."""
    if name not in KERNELS:
        raise KeyError(f"Unknown kernel: {name}")
    return KERNEL_GRAPHS_DIR / KERNELS[name]["graph"]


def get_metadata(name: str) -> dict:
    """Get metadata for a kernel."""
    if name not in KERNELS:
        raise KeyError(f"Unknown kernel: {name}")
    return KERNELS[name]
