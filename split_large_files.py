#!/usr/bin/env python3
"""Split large JSONL files into valid line-boundary parts.

This helper is local-only and does not run unless file paths are passed on the
command line. It refuses to overwrite existing split files. The original file is
kept unless --remove-original-after-split is explicitly provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def split_jsonl_file(
    file_path: Path,
    chunk_size_mb: int,
    remove_original: bool = False,
) -> list[Path]:
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix != ".jsonl":
        raise ValueError(f"Only .jsonl files are supported: {file_path}")

    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = file_path.stat().st_size
    if file_size <= chunk_size:
        print(f"{file_path} is already within the {chunk_size_mb} MB limit.")
        return []

    parts: list[Path] = []
    part_number = 1
    current_size = 0
    current_handle = None

    def open_next_part():
        nonlocal current_handle, current_size, part_number
        if current_handle is not None:
            current_handle.close()
        part_path = file_path.with_name(
            f"{file_path.stem}_part{part_number:02d}{file_path.suffix}"
        )
        if part_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing split file: {part_path}")
        current_handle = part_path.open("wb")
        parts.append(part_path)
        current_size = 0
        part_number += 1

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

    print(f"Split {file_path} into {len(parts)} valid JSONL part files.")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to split")
    parser.add_argument("--chunk-size-mb", type=int, default=1500)
    parser.add_argument(
        "--remove-original-after-split",
        action="store_true",
        help="Delete the original file only after all parts were written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for file_path in args.files:
        split_jsonl_file(
            file_path.expanduser(),
            chunk_size_mb=args.chunk_size_mb,
            remove_original=args.remove_original_after_split,
        )


if __name__ == "__main__":
    main()
