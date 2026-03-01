"""Kernel → Graph agent.

Reads a kernel source file, asks an LLM to produce a graph dict,
hydrates it, validates it, and loops on errors until valid.

Usage:
    python -m agent.run nvfp4_kernels/ptx/gemm/1.py
    OPENROUTER_API_KEY=sk-... python -m agent.run nvfp4_kernels/ptx/gemm/1.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from agent.llm import chat
from agent.prompt import SYSTEM_PROMPT, make_user_prompt, make_fix_prompt
from graph.interface.schema import graph_from_dict
from graph.interface.display import graph_string
from graph.state_machine.validator import validate_graph

OUT_DIR = Path(__file__).resolve().parent / "out"


def _log(log_path: Path, text: str) -> None:
    with open(log_path, "a") as f:
        f.write(text + "\n")


def run(kernel_path: str, max_rounds: int = 5, verbose: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # derive output names from kernel path: ptx/gemm/1.py -> ptx_gemm_1
    stem = kernel_path.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
    # strip leading nvfp4_kernels_ if present
    stem = re.sub(r"^nvfp4_kernels_", "", stem)
    log_path = OUT_DIR / f"{stem}.log"
    json_path = OUT_DIR / f"{stem}.graph.json"

    # clear previous log
    log_path.write_text("")

    def log(msg: str) -> None:
        print(msg)
        _log(log_path, msg)

    with open(kernel_path) as f:
        kernel_src = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(kernel_src)},
    ]

    for attempt in range(1, max_rounds + 1):
        log(f"\n--- round {attempt}/{max_rounds} ---")

        raw = chat(messages)
        messages.append({"role": "assistant", "content": raw})

        # always log full LLM response
        _log(log_path, f"\n[LLM response]\n{raw}\n")
        if verbose:
            print(raw[:2000])

        # extract JSON from response
        graph_dict = _extract_json(raw)
        if graph_dict is None:
            err = "Could not parse JSON from your response. Return ONLY a JSON object."
            log(f"  parse error: {err}")
            messages.append({"role": "user", "content": err})
            continue

        # sanitize common LLM mistakes before hydration
        graph_dict = _sanitize(graph_dict)

        # hydrate
        try:
            g = graph_from_dict(graph_dict)
        except Exception as e:
            err = f"graph_from_dict error: {e}"
            log(f"  hydration error: {err}")
            messages.append({"role": "user", "content": make_fix_prompt(err)})
            continue

        # validate
        try:
            validate_graph(g)
        except (ValueError, Exception) as e:
            err = str(e)
            log(f"  validation error: {err}")
            messages.append({"role": "user", "content": make_fix_prompt(err)})
            continue

        # success
        log("\n=== VALID GRAPH ===")
        gs = graph_string(g)
        log(gs)
        log(f"\nConverged in {attempt} round(s).")

        with open(json_path, "w") as f:
            json.dump(graph_dict, f, indent=2)
        log(f"Saved graph to {json_path}")
        log(f"Log at {log_path}")
        return

    log(f"\nFailed to produce a valid graph in {max_rounds} rounds.")
    # save last attempt anyway
    if graph_dict is not None:
        with open(json_path, "w") as f:
            json.dump(graph_dict, f, indent=2)
        log(f"Saved last (invalid) graph to {json_path}")
    sys.exit(1)


def _sanitize(graph_dict: dict) -> dict:
    """Fix common LLM mistakes in the graph dict."""
    for node in graph_dict.get("nodes", []):
        _sanitize_node(node)
    return graph_dict


def _sanitize_node(node: dict) -> None:
    """Strip leading dots from shape/tile values in op args."""
    if node.get("kind") == "Op":
        args = node.get("op_args", {})
        for key in ("shape", "tile", "num"):
            if isinstance(args.get(key), str) and args[key].startswith("."):
                args[key] = args[key].lstrip(".")
    for child in node.get("children", []):
        _sanitize_node(child)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of LLM response text."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Kernel → Graph agent")
    parser.add_argument("kernel", help="Path to kernel .py file")
    parser.add_argument("--rounds", type=int, default=5, help="Max LLM rounds")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.kernel, max_rounds=args.rounds, verbose=args.verbose)


if __name__ == "__main__":
    main()
