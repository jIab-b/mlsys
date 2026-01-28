import json
import os
from pathlib import Path

_CURRENT_WORKLOAD = None


def set_current_workload(workload: dict) -> None:
    global _CURRENT_WORKLOAD
    _CURRENT_WORKLOAD = workload


def get_current_workload() -> dict | None:
    return _CURRENT_WORKLOAD


def get_trace_root() -> Path | None:
    trace_root = os.environ.get("TRACE_OUT")
    if not trace_root:
        return None
    return Path(trace_root)


def append_trace_record(definition: str, record: dict, op_type: str) -> None:
    trace_root = get_trace_root()
    if trace_root is None:
        return
    traces_dir = trace_root / "traces" / op_type
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{definition}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
