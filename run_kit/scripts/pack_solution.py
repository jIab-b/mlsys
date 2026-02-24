"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine default source directory based on language
    if language == "triton":
        default_source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        default_source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    return _pack_solution_internal(
        default_source_dir=default_source_dir,
        entry_point=entry_point,
        language=language,
        solution_config=solution_config,
        output_path=output_path,
        source_path=None,
    )


def _resolve_source_path(source_path: Path | None) -> Path | None:
    if source_path is None:
        return None
    resolved = source_path.expanduser()
    if resolved.is_absolute():
        return resolved.resolve()

    # Prefer caller's current working directory for local pathing.
    cwd_candidate = (Path.cwd() / resolved).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback to run_kit project root for backward compatibility.
    return (PROJECT_ROOT / resolved).resolve()


def _pack_solution_internal(
    default_source_dir: Path,
    entry_point: str,
    language: str,
    solution_config: dict,
    output_path: Path | None,
    source_path: Path | None,
) -> Path:
    resolved_source_path = _resolve_source_path(source_path)
    tmp_dir_obj = None

    if resolved_source_path is None:
        source_dir = default_source_dir
    elif resolved_source_path.is_dir():
        source_dir = resolved_source_path
    elif resolved_source_path.is_file():
        # Pack only the specified file by staging it into an isolated temp source tree.
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="fib_pack_")
        source_dir = Path(tmp_dir_obj.name)
        entry_file = Path(entry_point.split("::")[0])
        staged_file = source_dir / entry_file
        staged_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_source_path, staged_file)
    else:
        raise FileNotFoundError(f"Source path not found: {resolved_source_path}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
    )

    # Pack the solution
    try:
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")
    if resolved_source_path is not None:
        print(f"  Source path: {resolved_source_path}")

    return output_path


def pack_solution_from_path(source_path: Path, output_path: Path = None) -> Path:
    """Pack solution files from an explicit source path (file or directory)."""
    config = load_config()
    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    if language == "triton":
        default_source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        default_source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    return _pack_solution_internal(
        default_source_dir=default_source_dir,
        entry_point=entry_point,
        language=language,
        solution_config=solution_config,
        output_path=output_path,
        source_path=source_path,
    )


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help=(
            "Optional source file or directory to pack instead of the default "
            "solution/<language> directory."
        ),
    )
    args = parser.parse_args()

    try:
        if args.source_path is None:
            pack_solution(args.output)
        else:
            pack_solution_from_path(args.source_path, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
