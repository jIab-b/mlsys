"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
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


def pack_solution(
    output_path: Path | None = None,
    file_base: str = "",
) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    spec = BuildSpec(language=language, target_hardware=["cuda"], entry_point=entry_point)

    pack_path = source_dir
    tmp_ctx = tempfile.TemporaryDirectory() if (file_base and language == "cuda") else None
    try:
        if file_base:
            if language != "cuda":
                raise ValueError("--file override is only supported for CUDA solutions")

            base = Path(file_base)
            if base.suffix in (".py", ".cu"):
                base = base.with_suffix("")

            py_src = source_dir / base.with_suffix(".py")
            cu_src = source_dir / base.with_suffix(".cu")
            if not py_src.is_file():
                raise FileNotFoundError(f"Missing override file: {py_src}")
            if not cu_src.is_file():
                raise FileNotFoundError(f"Missing override file: {cu_src}")

            tmpdir = Path(tmp_ctx.name)
            staged = tmpdir / "solution"
            staged.mkdir(parents=True, exist_ok=True)

            # Copy non-CUDA helper files and remap selected binding/kernel names.
            for path in source_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(source_dir)
                if "__pycache__" in rel.parts or path.suffix == ".cu":
                    continue
                dst = staged / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(path.read_bytes())

            (staged / "binding.py").write_bytes(py_src.read_bytes())
            (staged / "kernel.cu").write_bytes(cu_src.read_bytes())
            pack_path = staged

        solution = pack_solution_from_files(
            path=str(pack_path),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


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
        "--file",
        type=str,
        default="",
        help="CUDA override base name (maps <base>.py->binding.py and <base>.cu->kernel.cu)",
    )
    args = parser.parse_args()

    try:
        pack_solution(
            output_path=args.output,
            file_base=args.file,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
