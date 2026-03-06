"""Interface-layer wrapper for kernel->graph agent flow.

If legacy `agent` modules are available, this module reuses them.
Otherwise it falls back to an OpenRouter-backed client and local prompts.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from graph.interface.display import graph_string
from graph.interface.schema import graph_from_dict
from graph.ir import Graph
from graph.state_machine.sm100.validator import validate_graph as validate_graph_sm100

_INTERFACE_PROMPT_OVERLAY = """\
Additional required modeling constraints for descriptor legality:
- For tcgen05_mma ops, include idesc and smem_desc_a/smem_desc_b when available.
- For tcgen05_cp ops, include smem_desc when available.
- Descriptor objects should prefer explicit fields when known (raw/start_enc/ld_enc/sd_enc/base_offset/ld_mode/swizzle_code).
"""

_FALLBACK_SYSTEM_PROMPT = """\
Convert kernel source code into a graph JSON object for static/state-machine validation.

Return JSON only with top-level keys:
- buffers: [{name, space, shape, dtype, meta?}]
- barriers: [{name, scope?}]
- descriptors: [{name, buf?, meta?}]
- tmaps: [{name, ...metadata}]
- nodes: [node...]
- meta: {...}

Node schema:
- Generic node: {kind, args?, children?, meta?}
- Op node: {kind: \"Op\", op: \"<op_name>\", op_args: {...}}
- Raw node: {kind: \"Raw\", code: \"...\", meta?}

Requirements:
- Use valid JSON (double quotes, no comments, no trailing commas).
- Include tcgen descriptors in op_args when present in source:
  - tcgen05_mma: idesc, smem_desc_a, smem_desc_b
  - tcgen05_cp: smem_desc
- Prefer concrete numeric values if recoverable; otherwise provide conservative placeholders.
- Do not emit arithmetic expressions (e.g., `16 * 512`); emit final scalar numbers instead.
"""


@dataclass
class AgentInterfaceResult:
    ok: bool
    rounds: int
    graph: Optional[Graph] = None
    graph_dict: Optional[Dict[str, Any]] = None
    graph_text: str = ""
    errors: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)


def _try_load_legacy_agent_bindings() -> Optional[
    Tuple[Callable[[List[Dict[str, str]]], str], str, Callable[[str], str], Callable[[str], str]]
]:
    """Load legacy agent bindings if present."""
    for should_inject_repo_root in (False, True):
        try:
            if should_inject_repo_root:
                repo_root = Path(__file__).resolve().parents[2]
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))

            from agent.llm import chat as legacy_chat  # type: ignore[import-not-found]
            from agent.prompt import (  # type: ignore[import-not-found]
                SYSTEM_PROMPT as legacy_system_prompt,
                make_fix_prompt as legacy_make_fix_prompt,
                make_user_prompt as legacy_make_user_prompt,
            )

            return (
                legacy_chat,
                legacy_system_prompt,
                legacy_make_user_prompt,
                legacy_make_fix_prompt,
            )
        except Exception:
            continue
    return None


def _fallback_make_user_prompt(src: str) -> str:
    return "Kernel source:\n```text\n" + src + "\n```"


def _fallback_make_fix_prompt(error: str) -> str:
    return (
        "Previous response failed schema/validation checks.\n"
        f"Error: {error}\n"
        "Return a corrected JSON object only."
    )


def _openrouter_chat(messages: List[Dict[str, str]]) -> str:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if openrouter_key:
        model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
        url = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://local.graph.interface"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "graph-interface"),
        }
    elif openai_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        url = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
    else:
        raise RuntimeError(
            "No parser backend available: missing legacy `agent` package and both OPENROUTER_API_KEY/OPENAI_API_KEY are unset."
        )

    payload = {"model": model, "messages": messages, "temperature": 0}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")

    backend_name = "OpenRouter" if openrouter_key else "OpenAI"
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"{backend_name} HTTP error {e.code}: {body_text or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{backend_name} request failed: {e.reason}") from e

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"{backend_name} response missing choices.")
    content = choices[0].get("message", {}).get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(text_parts)
    raise RuntimeError(f"{backend_name} response missing message content.")


def run_agent_interface(
    kernel_path: str,
    *,
    max_rounds: int = 5,
    strict: bool = True,
    validate: bool = True,
    verbose: bool = False,
) -> AgentInterfaceResult:
    """Run kernel source -> graph generation through the existing agent stack."""
    src = Path(kernel_path).read_text()

    legacy = _try_load_legacy_agent_bindings()
    if legacy is None:
        chat_fn = _openrouter_chat
        system_prompt = _FALLBACK_SYSTEM_PROMPT
        make_user_prompt_fn = _fallback_make_user_prompt
        make_fix_prompt_fn = _fallback_make_fix_prompt
    else:
        chat_fn, system_prompt, make_user_prompt_fn, make_fix_prompt_fn = legacy

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt + "\n\n" + _INTERFACE_PROMPT_OVERLAY},
        {"role": "user", "content": make_user_prompt_fn(src)},
    ]
    errors: List[str] = []
    last_graph_dict: Optional[Dict[str, Any]] = None
    rounds = 0

    for attempt in range(1, max_rounds + 1):
        rounds = attempt
        raw = chat_fn(messages)
        messages.append({"role": "assistant", "content": raw})
        if verbose:
            print(raw[:2000])

        graph_dict = _extract_json(raw)
        if graph_dict is None:
            err = "Could not parse JSON from model response."
            errors.append(err)
            if strict:
                messages.append({"role": "user", "content": err + " Return ONLY a JSON object."})
                continue
            return AgentInterfaceResult(
                ok=False,
                rounds=rounds,
                graph_dict=last_graph_dict,
                errors=errors,
                messages=messages,
            )

        last_graph_dict = _sanitize(graph_dict)
        try:
            g = graph_from_dict(last_graph_dict)
        except Exception as e:  # hydration errors are surfaced as model-fix hints
            err = f"graph_from_dict error: {e}"
            errors.append(err)
            if strict:
                messages.append({"role": "user", "content": make_fix_prompt_fn(err)})
                continue
            return AgentInterfaceResult(
                ok=False,
                rounds=rounds,
                graph_dict=last_graph_dict,
                errors=errors,
                messages=messages,
            )

        if validate:
            try:
                validate_graph_sm100(g)
            except Exception as e:
                err = str(e)
                errors.append(err)
                if strict:
                    messages.append({"role": "user", "content": make_fix_prompt_fn(err)})
                    continue
                return AgentInterfaceResult(
                    ok=False,
                    rounds=rounds,
                    graph=g,
                    graph_dict=last_graph_dict,
                    graph_text=graph_string(g),
                    errors=errors,
                    messages=messages,
                )

        return AgentInterfaceResult(
            ok=True,
            rounds=rounds,
            graph=g,
            graph_dict=last_graph_dict,
            graph_text=graph_string(g),
            errors=errors,
            messages=messages,
        )

    return AgentInterfaceResult(
        ok=False,
        rounds=rounds,
        graph_dict=last_graph_dict,
        errors=errors,
        messages=messages,
    )


def _sanitize(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    for node in graph_dict.get("nodes", []):
        _sanitize_node(node)
    return graph_dict


def _sanitize_node(node: Dict[str, Any]) -> None:
    if node.get("kind") == "Op":
        args = node.get("op_args", {})
        for key in ("shape", "tile", "num"):
            if isinstance(args.get(key), str) and args[key].startswith("."):
                args[key] = args[key].lstrip(".")
    for child in node.get("children", []):
        _sanitize_node(child)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
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
                return _loads_json_with_fallback(text[start : i + 1])
    return None


def _loads_json_with_fallback(s: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Common model mistake: emits integer multiplication terms in JSON arrays/fields.
    s2 = re.sub(
        r"\b\d+(?:\s*\*\s*\d+)+\b",
        _eval_int_product,
        s,
    )
    if s2 == s:
        return None
    try:
        parsed = json.loads(s2)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _eval_int_product(match: re.Match[str]) -> str:
    expr = match.group(0)
    prod = 1
    for tok in re.split(r"\s*\*\s*", expr):
        prod *= int(tok)
    return str(prod)
