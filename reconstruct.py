#!/usr/bin/env python3
"""Reconstruct implementations from FlashInfer Trace definitions/solutions.

This script extracts PyTorch reference implementations from definitions
and CUDA/Triton source code from solutions.

Usage:
    python reconstruct.py                           # Use defaults (DSA kernels)
    python reconstruct.py --id dsa_topk_indexer_fp8_h64_d128_topk256_ps64
    python reconstruct.py --id dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64
    python reconstruct.py --list                    # List available definitions
    python reconstruct.py --solution <name>         # Reconstruct a solution
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FLASHINFER_TRACE = PROJECT_ROOT / "eval_suite" / "flashinfer_trace"
DEFINITIONS_DIR = FLASHINFER_TRACE / "definitions"
SOLUTIONS_DIR = FLASHINFER_TRACE / "solutions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reconstructed"

# Default IDs: DSA sparse index + attention kernels
DEFAULT_DEFINITION_IDS = [
    "dsa_topk_indexer_fp8_h64_d128_topk256_ps64",
    "dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64",
]


def find_definition(definition_id: str) -> Path | None:
    """Find a definition JSON file by ID (name)."""
    for json_file in DEFINITIONS_DIR.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if data.get("name") == definition_id:
                return json_file
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def find_solution(solution_id: str) -> Path | None:
    """Find a solution JSON file by ID (name)."""
    if not SOLUTIONS_DIR.exists():
        return None
    for json_file in SOLUTIONS_DIR.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if data.get("name") == solution_id:
                return json_file
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def list_definitions() -> list[dict]:
    """List all available definitions."""
    definitions = []
    for json_file in DEFINITIONS_DIR.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            definitions.append({
                "name": data.get("name", "unknown"),
                "op_type": data.get("op_type", "unknown"),
                "description": data.get("description", "")[:80],
                "path": str(json_file.relative_to(FLASHINFER_TRACE)),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(definitions, key=lambda x: (x["op_type"], x["name"]))


def list_solutions() -> list[dict]:
    """List all available solutions."""
    if not SOLUTIONS_DIR.exists():
        return []
    solutions = []
    for json_file in SOLUTIONS_DIR.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            solutions.append({
                "name": data.get("name", "unknown"),
                "definition": data.get("definition", "unknown"),
                "author": data.get("author", "unknown"),
                "language": data.get("spec", {}).get("language", "unknown"),
                "path": str(json_file.relative_to(FLASHINFER_TRACE)),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(solutions, key=lambda x: x["name"])


def reconstruct_definition(definition_id: str, output_dir: Path) -> dict:
    """Reconstruct a definition's reference implementation.

    Returns dict with:
        - reference_code: str (the PyTorch reference implementation)
        - signature: dict (input/output specs)
        - constants: dict (fixed axis values)
        - constraints: list (invariants)
    """
    json_path = find_definition(definition_id)
    if json_path is None:
        raise ValueError(f"Definition not found: {definition_id}")

    data = json.loads(json_path.read_text())

    # Create output directory
    def_output = output_dir / "definitions" / definition_id
    def_output.mkdir(parents=True, exist_ok=True)

    # Write reference implementation
    reference_code = data.get("reference", "")
    ref_path = def_output / "reference.py"
    ref_path.write_text(reference_code)

    # Write full definition JSON for inspection
    (def_output / "definition.json").write_text(json.dumps(data, indent=2))

    # Extract structured info
    result = {
        "name": data.get("name"),
        "op_type": data.get("op_type"),
        "reference_path": str(ref_path),
        "inputs": data.get("inputs", {}),
        "outputs": data.get("outputs", {}),
        "constants": {
            name: ax["value"]
            for name, ax in data.get("axes", {}).items()
            if ax.get("type") == "const"
        },
        "variable_axes": [
            name for name, ax in data.get("axes", {}).items()
            if ax.get("type") == "var"
        ],
        "constraints": data.get("constraints", []),
    }

    # Write a summary
    summary_path = def_output / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Definition: {result['name']}\n")
        f.write(f"Op Type: {result['op_type']}\n")
        f.write(f"\nConstants:\n")
        for k, v in result["constants"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nVariable Axes: {', '.join(result['variable_axes'])}\n")
        f.write(f"\nInputs:\n")
        for name, spec in result["inputs"].items():
            f.write(f"  {name}: shape={spec.get('shape')}, dtype={spec.get('dtype')}\n")
        f.write(f"\nOutputs:\n")
        for name, spec in result["outputs"].items():
            f.write(f"  {name}: shape={spec.get('shape')}, dtype={spec.get('dtype')}\n")
        if result["constraints"]:
            f.write(f"\nConstraints:\n")
            for c in result["constraints"]:
                f.write(f"  {c}\n")

    print(f"  Reconstructed: {def_output}")
    return result


def reconstruct_solution(solution_id: str, output_dir: Path) -> dict:
    """Reconstruct a solution's source files.

    Returns dict with:
        - sources: list of {path, content}
        - entry_point: str
        - spec: dict
    """
    json_path = find_solution(solution_id)
    if json_path is None:
        raise ValueError(f"Solution not found: {solution_id}")

    data = json.loads(json_path.read_text())

    # Create output directory
    sol_output = output_dir / "solutions" / solution_id
    sol_output.mkdir(parents=True, exist_ok=True)

    # Write all source files
    sources = data.get("sources", [])
    for src in sources:
        src_path = sol_output / src["path"]
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text(src["content"])

    # Write full solution JSON
    (sol_output / "solution.json").write_text(json.dumps(data, indent=2))

    result = {
        "name": data.get("name"),
        "definition": data.get("definition"),
        "author": data.get("author"),
        "entry_point": data.get("spec", {}).get("entry_point"),
        "language": data.get("spec", {}).get("language"),
        "target_hardware": data.get("spec", {}).get("target_hardware", []),
        "source_paths": [src["path"] for src in sources],
    }

    print(f"  Reconstructed: {sol_output}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct implementations from FlashInfer Trace"
    )
    parser.add_argument(
        "--id", "-i",
        action="append",
        dest="ids",
        help="Definition ID(s) to reconstruct (can specify multiple)",
    )
    parser.add_argument(
        "--solution", "-s",
        action="append",
        dest="solutions",
        help="Solution ID(s) to reconstruct (can specify multiple)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available definitions and solutions",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reconstruct all definitions",
    )

    args = parser.parse_args()

    if args.list:
        print("Available Definitions:")
        print("-" * 80)
        for d in list_definitions():
            print(f"  {d['op_type']:20} {d['name']}")
            if d['description']:
                print(f"                       {d['description']}")
        print()
        solutions = list_solutions()
        if solutions:
            print("Available Solutions:")
            print("-" * 80)
            for s in solutions:
                print(f"  {s['name']:40} ({s['language']}, by {s['author']})")
        return 0

    # Determine what to reconstruct
    definition_ids = args.ids or []
    solution_ids = args.solutions or []

    if args.all:
        definition_ids = [d["name"] for d in list_definitions()]
    elif not definition_ids and not solution_ids:
        # Default: DSA kernels
        definition_ids = DEFAULT_DEFINITION_IDS

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Reconstruct definitions
    if definition_ids:
        print(f"Reconstructing {len(definition_ids)} definition(s)...")
        for def_id in definition_ids:
            try:
                reconstruct_definition(def_id, output_dir)
            except ValueError as e:
                print(f"  ERROR: {e}", file=sys.stderr)

    # Reconstruct solutions
    if solution_ids:
        print(f"\nReconstructing {len(solution_ids)} solution(s)...")
        for sol_id in solution_ids:
            try:
                reconstruct_solution(sol_id, output_dir)
            except ValueError as e:
                print(f"  ERROR: {e}", file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
