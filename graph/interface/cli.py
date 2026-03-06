"""Public interface CLI for source->graph and selected SM100 checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from graph.interface.agent_interface import run_agent_interface
from graph.interface.schema import graph_from_dict
from graph.ir import Graph, Node
from graph.ptx_ops.spec import _canonical_op_name
from graph.ptx_ops.static import validate_ptx_op
from graph.state_machine.sm100.idesc import validate_tcgen_descriptor_layout
from graph.state_machine.sm100.validator import validate_graph as validate_graph_sm100


def _iter_executed_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        if node.kind == "For":
            iters = int(node.args.get("iters", 0))
            for _ in range(iters):
                yield from _iter_executed_nodes(node.children)
            continue
        if node.kind == "If":
            then_node = next((c for c in node.children if c.kind == "Then"), None)
            else_node = next((c for c in node.children if c.kind == "Else"), None)
            if then_node:
                yield from _iter_executed_nodes(then_node.children)
            if else_node:
                yield from _iter_executed_nodes(else_node.children)
            continue
        if node.children:
            yield from _iter_executed_nodes(node.children)
        yield node


def _run_idesc_checks(g: Graph) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    mma_issues = 0
    for node in _iter_executed_nodes(g.nodes):
        if node.kind != "Op":
            continue
        op_name = str(node.args.get("op", ""))
        op_args = dict(node.args.get("op_args", {}))
        canonical = _canonical_op_name(op_name)

        if canonical in {"tcgen05_mma", "tcgen05_cp"}:
            mma_issues += 1

        try:
            validate_ptx_op(op_name, op_args)
        except Exception as e:
            findings.append(
                {
                    "check": "idesc",
                    "op": op_name,
                    "message": f"static descriptor legality: {e}",
                }
            )
            continue

        if canonical in {"tcgen05_mma", "tcgen05_cp"}:
            try:
                validate_tcgen_descriptor_layout(canonical, op_name, op_args, g)
            except Exception as e:
                findings.append(
                    {
                        "check": "idesc",
                        "op": op_name,
                        "message": f"dynamic descriptor/layout legality: {e}",
                    }
                )

    return findings


def _load_graph(path: str, *, no_strict: bool, rounds: int, verbose: bool) -> Graph:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        return graph_from_dict(data)

    res = run_agent_interface(
        path,
        max_rounds=rounds,
        strict=not no_strict,
        validate=False,
        verbose=verbose,
    )
    if res.graph is None:
        err = res.errors[-1] if res.errors else "failed to build graph from source"
        raise RuntimeError(err)
    return res.graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph interface CLI")
    parser.add_argument("--path_to_src_file", required=False, help="Kernel source file or graph JSON path")
    parser.add_argument("--no_strict", action="store_true", help="Return full results regardless of check failure")
    parser.add_argument("-o", "--output", help="Write JSON results to this file")
    parser.add_argument(
        "--checks",
        "--checsk",
        default="all",
        help="Comma-separated checks to run. Supported: all,idesc",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Max agent rounds for source parsing")
    parser.add_argument("--verbose", action="store_true", help="Verbose agent output")
    args, unknown = parser.parse_known_args()

    inferred_path = None
    leftovers = []
    for tok in unknown:
        if tok.startswith("--") and "/" in tok[2:] and inferred_path is None:
            inferred_path = tok[2:]
            continue
        leftovers.append(tok)
    if leftovers:
        parser.error(f"unrecognized arguments: {' '.join(leftovers)}")

    if not args.path_to_src_file and inferred_path:
        args.path_to_src_file = inferred_path
    if not args.path_to_src_file:
        parser.error("--path_to_src_file is required")

    checks = [c.strip().lower() for c in args.checks.split(",") if c.strip()]
    if not checks:
        checks = ["all"]
    unsupported = [c for c in checks if c not in {"all", "idesc"}]
    if unsupported:
        print(json.dumps({"status": "error", "message": f"unsupported checks: {unsupported}"}, indent=2))
        sys.exit(2)

    try:
        g = _load_graph(
            args.path_to_src_file,
            no_strict=args.no_strict,
            rounds=args.rounds,
            verbose=args.verbose,
        )
    except Exception as e:
        print(json.dumps({"status": "error", "phase": "parse", "message": str(e)}, indent=2))
        sys.exit(1)

    findings: List[Dict[str, Any]] = []
    checks_run: List[str] = []

    if "all" in checks:
        checks_run.append("all")
        try:
            validate_graph_sm100(g)
        except Exception as e:
            findings.append({"check": "all", "message": str(e)})
    else:
        if "idesc" in checks:
            checks_run.append("idesc")
            findings.extend(_run_idesc_checks(g))

    failed = len(findings) > 0
    out = {
        "status": "fail" if failed else "pass",
        "strict": not args.no_strict,
        "checks_run": checks_run,
        "findings": findings,
        "meta": {
            "nodes": len(g.nodes),
            "buffers": len(g.buffers),
            "barriers": len(g.barriers),
            "descriptors": len(g.descriptors),
            "tmaps": len(g.tmaps),
        },
    }
    out_text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(out_text + "\n")
    else:
        print(out_text)

    if failed and not args.no_strict:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
