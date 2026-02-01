CHECK = "\u2705"
CROSS = "\u274c"
STOPWATCH = "\u23f1"
ZAP = "\u26a1"
SNAIL = "\U0001F40C"
MICRO = "\u00b5"


def _reorder_spec(spec: str) -> str:
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    kv = {}
    for part in parts:
        key, sep, value = part.partition(":")
        if sep:
            kv[key.strip()] = value.strip()
    if not kv:
        return spec.strip()
    order = ["batch", "num_pages", "seq_len", "seed"]
    ordered = [f"{k}: {kv[k]}" for k in order if k in kv]
    for k in sorted(k for k in kv if k not in order):
        ordered.append(f"{k}: {kv[k]}")
    return "; ".join(ordered)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_popcorn(log_text: str) -> dict:
    data: dict = {"benchmarks": {}, "tests": {}}
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key in {"compile", "compile.error", "check", "benchmark-count", "test-count"}:
            data[key.replace(".", "_")] = value
            continue
        if key.startswith("benchmark."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                data["benchmarks"].setdefault(int(parts[1]), {})[parts[2]] = value
            continue
        if key.startswith("test."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                data["tests"].setdefault(int(parts[1]), {})[parts[2]] = value
    return data


def _format_benchmarks(benchmarks: dict[int, dict[str, str]]) -> list[str]:
    lines = ["## Benchmarks:", "```"]
    items = sorted(benchmarks.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))
        mean = _to_float(info.get("mean"))
        err = _to_float(info.get("err"))
        best = _to_float(info.get("best"))
        worst = _to_float(info.get("worst"))
        if None not in (mean, err, best, worst):
            lines.append(f" {STOPWATCH} {mean / 1000:.1f} \u00b1 {err / 1000:.2f} {MICRO}s")
            lines.append(f" {ZAP} {best / 1000:.1f} {MICRO}s {SNAIL} {worst / 1000:.1f} {MICRO}s")
        if info.get("status") == "fail" or info.get("error"):
            lines.append(f" {CROSS} {info.get('error', 'failed')}")
        if idx != items[-1][0]:
            lines.append("")
    lines.append("```")
    return lines


def _format_tests(tests: dict[int, dict[str, str]]) -> list[str]:
    lines = ["## Tests:", "```"]
    items = sorted(tests.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))
        status = info.get("status", "")
        if status == "pass":
            lines.append(f" {CHECK} pass")
        elif status == "fail":
            lines.append(f" {CROSS} fail")
        if info.get("error"):
            lines.append(f"   {info['error']}")
        if idx != items[-1][0]:
            lines.append("")
    lines.append("```")
    return lines


def _format_program_output(stdout: str, stderr: str) -> list[str]:
    combined = stdout or ""
    if stderr:
        combined = (combined.rstrip("\n") + "\n--- stderr ---\n" if combined else "--- stderr ---\n") + stderr
    lines = ["## Program stdout:", "```"]
    if combined:
        lines.extend(combined.rstrip("\n").splitlines())
    lines.append("```")
    return lines


def _phase_label(mode: str, data: dict) -> str:
    if data.get("benchmarks"):
        return "Benchmarking"
    if data.get("tests"):
        return "Testing"
    return "Profiling" if mode == "profile" else "Testing" if mode == "test" else "Run"


def _build_formatted_output(data: dict, stdout: str, stderr: str, system: dict | None, mode: str) -> str:
    compile_status = data.get("compile")
    check_status = data.get("check")
    success = check_status == "pass" and compile_status != "fail"
    lines = [f"\"**Modal {CHECK if success else CROSS} {'success' if success else 'failure'}**"]
    if compile_status:
        lines.append(
            f"> {CHECK if compile_status == 'pass' else CROSS} "
            f"Compilation {'successful' if compile_status == 'pass' else 'failed'}"
        )
    if check_status:
        phase = _phase_label(mode, data)
        lines.append(
            f"> {CHECK if check_status == 'pass' else CROSS} "
            f"{phase} {'successful' if check_status == 'pass' else 'failed'}"
        )
    lines.append("")
    if system:
        lines.extend([
            "Running on:",
            f"* GPU: `{system.get('gpu', 'Unknown')}`",
            f"* CPU: `{system.get('cpu', 'Unknown')}`",
            f"* Device count: `{system.get('device_count', 'Unknown')}`",
            f"* Runtime: `{system.get('runtime', 'Unknown')}`",
            f"* Platform: `{system.get('platform', 'Unknown')}`",
            f"* Torch: `{system.get('torch', 'Unknown')}`",
            f"* Hostname: `{system.get('hostname', 'Unknown')}`",
            "",
            "",
        ])
    if data.get("benchmarks"):
        lines.extend(_format_benchmarks(data["benchmarks"]))
        lines.append("")
    elif data.get("tests"):
        lines.extend(_format_tests(data["tests"]))
        lines.append("")
    lines.extend(_format_program_output(stdout, stderr))
    return "\n".join(lines) + "\""


def format_result(result, mode: str) -> str:
    if isinstance(result, str) and result.lstrip().startswith('"**'):
        return result
    if isinstance(result, str):
        data = _parse_popcorn(result)
        return _build_formatted_output(data, "", "", None, mode)
    popcorn = result.get("popcorn", "")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    system = result.get("system")
    if popcorn.lstrip().startswith('"**'):
        return popcorn
    data = _parse_popcorn(popcorn)
    return _build_formatted_output(data, stdout, stderr, system, result.get("mode", mode))
