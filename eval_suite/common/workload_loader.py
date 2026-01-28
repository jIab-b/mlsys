"""Load and sample workloads from FlashInfer Trace datasets."""
import json
import random
from pathlib import Path


FLASHINFER_TRACE = Path(__file__).parent.parent / "flashinfer_trace"
WORKLOADS_DIR = FLASHINFER_TRACE / "workloads"
DEFINITIONS_DIR = FLASHINFER_TRACE / "definitions"


def load_workloads(definition_name: str, op_type: str | None = None) -> list[dict]:
    """Load all workloads for a given definition.

    Args:
        definition_name: The definition name to load workloads for
        op_type: Optional op_type subdirectory to search in

    Returns:
        List of workload dicts with 'axes' and 'inputs' keys
    """
    workloads = []

    # Search pattern
    if op_type:
        search_dirs = [WORKLOADS_DIR / op_type]
    else:
        search_dirs = list(WORKLOADS_DIR.iterdir()) if WORKLOADS_DIR.exists() else []

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        jsonl_path = search_dir / f"{definition_name}.jsonl"
        if jsonl_path.exists():
            with jsonl_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "workload" in record:
                            workloads.append(record["workload"])
                    except json.JSONDecodeError:
                        continue

    return workloads


def sample_workloads(
    definition_name: str,
    op_type: str | None = None,
    count: int = 3,
    seed: int | None = None,
) -> list[dict]:
    """Sample random workloads for a definition.

    Args:
        definition_name: The definition name
        op_type: Optional op_type subdirectory
        count: Number of workloads to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled workload dicts
    """
    workloads = load_workloads(definition_name, op_type)
    if not workloads:
        return []

    if seed is not None:
        random.seed(seed)

    if len(workloads) <= count:
        return workloads

    return random.sample(workloads, count)


def workload_to_test_spec(workload: dict, seed: int = 42) -> str:
    """Convert a workload dict to a test spec string.

    The test spec format is: "key1: value1; key2: value2; ..."
    Only axes (variable dimensions) are included.

    Args:
        workload: Workload dict with 'axes' key
        seed: Seed value to include in the spec

    Returns:
        Test spec string compatible with eval_base.get_test_cases
    """
    axes = workload.get("axes", {})
    parts = []

    for key, value in sorted(axes.items()):
        parts.append(f"{key}: {value}")

    parts.append(f"seed: {seed}")
    return "; ".join(parts)


def sample_test_specs(
    definition_name: str,
    op_type: str | None = None,
    count: int = 3,
    seed: int = 42,
) -> list[str]:
    """Sample workloads and convert to test spec strings.

    Args:
        definition_name: The definition name
        op_type: Optional op_type subdirectory
        count: Number of specs to generate
        seed: Random seed

    Returns:
        List of test spec strings
    """
    workloads = sample_workloads(definition_name, op_type, count, seed)
    return [
        workload_to_test_spec(w, seed=seed + i)
        for i, w in enumerate(workloads)
    ]


def get_similar_workloads(
    target_op_type: str,
    fallback_op_types: list[str] | None = None,
    count: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Get workloads from similar definitions when exact match not available.

    For DSA (dsa_paged), falls back to MLA (mla_paged) which has similar structure.

    Args:
        target_op_type: The target op_type
        fallback_op_types: List of fallback op_types to try
        count: Number of workloads to sample
        seed: Random seed

    Returns:
        List of workload dicts
    """
    # Default fallbacks for common op types
    if fallback_op_types is None:
        fallback_op_types = {
            "dsa_paged": ["mla_paged", "gqa_paged"],
            "sparse_attention": ["mla_paged", "gqa_paged"],
        }.get(target_op_type, [])

    # Try target op_type first
    op_type_dir = WORKLOADS_DIR / target_op_type
    if op_type_dir.exists():
        for jsonl_file in op_type_dir.glob("*.jsonl"):
            definition_name = jsonl_file.stem
            workloads = sample_workloads(definition_name, target_op_type, count, seed)
            if workloads:
                return workloads

    # Try fallbacks
    for fallback in fallback_op_types:
        fallback_dir = WORKLOADS_DIR / fallback
        if fallback_dir.exists():
            for jsonl_file in fallback_dir.glob("*.jsonl"):
                definition_name = jsonl_file.stem
                workloads = sample_workloads(definition_name, fallback, count, seed)
                if workloads:
                    return workloads

    return []


def generate_sparse_attention_specs(count: int = 3, seed: int = 42) -> list[str]:
    """Generate test specs for sparse_attention using MLA workload patterns.

    The sparse_attention task uses batch, num_pages, seq_len as variable axes.
    We sample from MLA workloads and map their axes.

    Args:
        count: Number of specs to generate
        seed: Random seed

    Returns:
        List of test spec strings
    """
    # Get MLA workloads as reference for realistic axis values
    workloads = get_similar_workloads("sparse_attention", count=count * 2, seed=seed)

    if not workloads:
        # Fallback to reasonable defaults if no workloads found
        return [
            f"batch: 1; num_pages: 16; seq_len: 512; seed: {seed}",
            f"batch: 4; num_pages: 32; seq_len: 1024; seed: {seed + 1}",
            f"batch: 8; num_pages: 64; seq_len: 2048; seed: {seed + 2}",
        ][:count]

    random.seed(seed)
    sampled = random.sample(workloads, min(count, len(workloads)))

    specs = []
    for i, wl in enumerate(sampled):
        axes = wl.get("axes", {})
        # Map MLA axes to sparse_attention axes
        batch = axes.get("batch_size", 1)
        num_pages = min(axes.get("num_pages", 16), 128)  # Cap for memory
        # Derive seq_len from num_kv_indices or reasonable default
        seq_len = min(axes.get("num_kv_indices", 512), num_pages * 64)

        spec = f"batch: {batch}; num_pages: {num_pages}; seq_len: {seq_len}; seed: {seed + i}"
        specs.append(spec)

    return specs


def list_available_workloads() -> dict[str, list[str]]:
    """List all available workloads organized by op_type.

    Returns:
        Dict mapping op_type to list of definition names
    """
    result = {}
    if not WORKLOADS_DIR.exists():
        return result

    for op_dir in WORKLOADS_DIR.iterdir():
        if not op_dir.is_dir():
            continue
        definitions = []
        for jsonl_file in op_dir.glob("*.jsonl"):
            definitions.append(jsonl_file.stem)
        if definitions:
            result[op_dir.name] = sorted(definitions)

    return result
