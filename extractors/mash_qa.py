from __future__ import annotations

from collections.abc import Iterator
import json
from hashlib import sha256
from pathlib import Path

import click
from tqdm import tqdm

from entities import Pass1Record

DATASET_DIR = Path("datasets.original/mash_qa")
DATA_FILES = [
    DATASET_DIR / "mashqa_data" / "train_webmd_squad_v2_full.json",
    DATASET_DIR / "mashqa_data" / "val_webmd_squad_v2_full.json",
    DATASET_DIR / "mashqa_data" / "test_webmd_squad_v2_full.json",
    DATASET_DIR / "mashqa_data" / "train_webmd_squad_v2_consec.json",
    DATASET_DIR / "mashqa_data" / "val_webmd_squad_v2_consec.json",
    DATASET_DIR / "mashqa_data" / "test_webmd_squad_v2_consec.json",
]

assert DATASET_DIR.exists(), f"Missing dataset directory: {DATASET_DIR}"


def _record(question: str) -> Pass1Record:
    question = question.strip()
    return Pass1Record(id=sha256(question.encode("utf-8")).hexdigest(), question=question)


def extract() -> Iterator[Pass1Record]:
    for path in tqdm(DATA_FILES, desc="mash_qa files", unit="file"):
        assert path.is_file(), f"missing source file: {path}"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload, dict), f"unexpected JSON structure in {path}"
        entries = payload["data"]
        for entry in tqdm(entries, desc=path.name, unit="entry", leave=False):
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    yield _record(qa["question"])


def _dry_run() -> None:
    size_bytes = sum(path.stat().st_size for path in DATA_FILES if path.is_file())
    count = 0
    for path in DATA_FILES:
        assert path.is_file(), f"missing source file: {path}"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload, dict), f"unexpected JSON structure in {path}"
        count += sum(len(paragraph["qas"]) for entry in payload["data"] for paragraph in entry["paragraphs"])
    click.echo(f"location={DATASET_DIR}")
    click.echo(f"size_bytes={size_bytes}")
    click.echo(f"count={count}")


@click.command()
@click.option("--dry-run", is_flag=True, help="Print dataset location and size/count, then exit.")
@click.option("--limit", type=int, default=None, help="Emit at most N records.")
def cli(dry_run: bool, limit: int | None) -> None:
    if dry_run:
        _dry_run()
        return
    if limit is not None and limit < 1:
        raise click.BadParameter("--limit must be at least 1")
    for index, record in enumerate(extract(), start=1):
        click.echo(record.model_dump_json())
        if limit is not None and index >= limit:
            break


if __name__ == "__main__":
    cli()
