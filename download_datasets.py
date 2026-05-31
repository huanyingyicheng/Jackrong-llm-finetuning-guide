#!/usr/bin/env python3
"""Download Jackrong datasets into the local High-fidelity Dataset folder.

This helper is intentionally local-only: it never stages, commits, pushes, or
publishes repository contents. Large JSONL splitting is optional and preserves
line boundaries so each generated part remains valid JSONL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_TARGET_DIR = Path(__file__).resolve().parent / "High-fidelity Dataset"


def split_jsonl_on_line_boundaries(
    file_path: Path,
    chunk_size_mb: int,
    remove_original: bool = False,
) -> list[Path]:
    """Split a JSONL file without cutting through records."""

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    chunk_size = chunk_size_mb * 1024 * 1024
    if file_path.stat().st_size <= chunk_size:
        return []

    parts: list[Path] = []
    part_number = 1
    current_size = 0
    current_handle = None

    def open_next_part():
        nonlocal part_number, current_size, current_handle
        if current_handle is not None:
            current_handle.close()
        part_path = file_path.with_name(
            f"{file_path.stem}_part{part_number:02d}{file_path.suffix}"
        )
        if part_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing split file: {part_path}")
        current_handle = part_path.open("wb")
        parts.append(part_path)
        part_number += 1
        current_size = 0

    try:
        open_next_part()
        with file_path.open("rb") as source:
            for line in source:
                if current_size and current_size + len(line) > chunk_size:
                    open_next_part()
                current_handle.write(line)
                current_size += len(line)
    finally:
        if current_handle is not None:
            current_handle.close()

    if remove_original:
        file_path.unlink()

    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default="Jackrong", help="Hugging Face dataset author")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument(
        "--split-large-jsonl",
        action="store_true",
        help="Split downloaded JSONL files larger than --chunk-size-mb on line boundaries",
    )
    parser.add_argument("--chunk-size-mb", type=int, default=1500)
    parser.add_argument(
        "--remove-original-after-split",
        action="store_true",
        help="Delete the original large JSONL after successful line-safe splitting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    datasets = list(api.list_datasets(author=args.author))
    print(f"Found {len(datasets)} datasets from {args.author}")

    for index, dataset in enumerate(datasets, 1):
        dataset_name = dataset.id.split("/")[-1]
        local_dir = target_dir / dataset_name
        print(f"[{index}/{len(datasets)}] Downloading {dataset.id} -> {local_dir}")

        snapshot_download(
            repo_id=dataset.id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )

        if args.split_large_jsonl:
            for jsonl_path in local_dir.rglob("*.jsonl"):
                parts = split_jsonl_on_line_boundaries(
                    jsonl_path,
                    chunk_size_mb=args.chunk_size_mb,
                    remove_original=args.remove_original_after_split,
                )
                if parts:
                    print(f"  Split {jsonl_path.name} into {len(parts)} line-safe parts")

    print(f"Downloads complete. Files are in: {target_dir}")
    print("No git add, commit, push, or remote publish operation was performed.")


if __name__ == "__main__":
    main()
