"""Thin OpenRouter chat wrapper."""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error

MODEL = "qwen/qwen3-next-80b-a3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 5
RETRY_BACKOFF = [10, 20, 40, 60, 90]


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY env var. "
            "e.g. OPENROUTER_API_KEY=sk-or-... python -m agent.run ..."
        )
    return key


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    """Send messages to OpenRouter, return assistant text."""
    key = _get_api_key()
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }).encode()

    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mlsys",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"  rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter API {e.code}: {err_body}") from e

    raise RuntimeError("Exhausted retries")
