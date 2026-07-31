#!/usr/bin/env python3
"""Stream-shuffle a large HF dataset into a small text-only local subset.

The output layout is intentionally plain:

    output_dir/
      train.parquet
      test.parquet

That directory can be consumed by `datasets.load_dataset(output_dir)`, which is
the loading style used by this project.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


SCHEMA = pa.schema([("text", pa.string())])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a shuffled text-only subset from a streaming Hugging Face dataset."
    )
    parser.add_argument("dataset", help="Dataset name or path passed to datasets.load_dataset().")
    parser.add_argument("--name", default=None, help="Optional dataset config/name.")
    parser.add_argument("--split", default="train", help="Source split to stream from.")
    parser.add_argument("--num-samples", type=int, required=True, help="Total rows to write.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.001,
        help="Fraction of rows used for the test split when --test-samples is omitted.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Exact number of rows for the test split. Overrides --test-ratio.",
    )
    parser.add_argument("--text-column", default="text", help="Source column to save as text.")
    parser.add_argument("--output-dir", default=None, help="Output dataset directory.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache dir passed to load_dataset(). Defaults to HF settings.",
    )
    parser.add_argument("--data-dir", default=None, help="Optional data_dir for load_dataset().")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10000,
        help="Streaming shuffle buffer size. Use 0 to disable shuffle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per Parquet write.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dir.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Forwarded to load_dataset().")
    parser.add_argument("--log-every", type=int, default=10000, help="Progress logging interval.")
    return parser.parse_args()


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def default_output_dir(args: argparse.Namespace, test_samples: int) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    parts = [args.dataset]
    if args.name:
        parts.append(args.name)
    parts.append(args.split)
    slug = safe_slug("__".join(parts))
    suffix = f"n{args.num_samples}_test{test_samples}_seed{args.seed}"
    return hf_home / "datasets" / "stream_sample" / f"{slug}__{suffix}"


def split_sizes(num_samples: int, test_ratio: float, test_samples: int | None) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if test_samples is None:
        if test_ratio <= 0 or num_samples == 1:
            test_samples = 0
        else:
            test_samples = max(1, int(math.ceil(num_samples * test_ratio)))
    if test_samples < 0 or test_samples >= num_samples:
        raise ValueError("--test-samples must be in [0, --num-samples).")
    return num_samples - test_samples, test_samples


def normalize_text(example: dict[str, Any], text_column: str) -> str | None:
    value = example.get(text_column)
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        text = "\n".join(str(item) for item in value if item is not None)
    else:
        text = str(value)
    return text if text.strip() else None


class ParquetTextWriter:
    def __init__(self, path: Path, batch_size: int):
        self.path = path
        self.batch_size = batch_size
        self.count = 0
        self._buffer: list[str] = []
        self._writer: pq.ParquetWriter | None = None

    def write(self, text: str) -> None:
        self._buffer.append(text)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pydict({"text": self._buffer}, schema=SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, SCHEMA, compression="zstd")
        self._writer.write_table(table)
        self.count += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            pq.write_table(pa.Table.from_pydict({"text": []}, schema=SCHEMA), self.path)
        else:
            self._writer.close()


def stream_examples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    kwargs = {
        "path": args.dataset,
        "name": args.name,
        "split": args.split,
        "streaming": True,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    dataset = load_dataset(**kwargs)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return dataset


def prepare_output(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=False)
    return tmp_path


def main() -> None:
    args = parse_args()
    train_target, test_target = split_sizes(args.num_samples, args.test_ratio, args.test_samples)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(args, test_target)
    tmp_dir = prepare_output(output_dir, args.overwrite)

    train_writer = ParquetTextWriter(tmp_dir / "train.parquet", args.batch_size)
    test_writer = ParquetTextWriter(tmp_dir / "test.parquet", args.batch_size)
    skipped = 0

    try:
        for example in stream_examples(args):
            text = normalize_text(example, args.text_column)
            if text is None:
                skipped += 1
                continue

            if train_writer.count + len(train_writer._buffer) < train_target:
                train_writer.write(text)
            elif test_writer.count + len(test_writer._buffer) < test_target:
                test_writer.write(text)
            else:
                break

            written = (
                train_writer.count
                + len(train_writer._buffer)
                + test_writer.count
                + len(test_writer._buffer)
            )
            if args.log_every > 0 and written % args.log_every == 0:
                print(f"written={written} skipped={skipped}", flush=True)

        train_writer.close()
        test_writer.close()

        written = train_writer.count + test_writer.count
        if written < args.num_samples:
            raise RuntimeError(
                f"Only wrote {written} rows; source exhausted or too many rows lacked {args.text_column!r}."
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"saved: {output_dir}")
    print(f"train: {train_writer.count} rows")
    print(f"test: {test_writer.count} rows")
    print(f"load with: load_dataset({str(output_dir)!r})")


if __name__ == "__main__":
    main()
