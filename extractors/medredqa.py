from __future__ import annotations

from collections.abc import Iterator
import csv
from hashlib import sha256
from pathlib import Path

import click
from tqdm import tqdm

from entities import Pass1Record

DATASET_DIR = Path("datasets.original/medredqa/medredqa")
DATA_FILES = [
    DATASET_DIR / "medredqa_train.csv",
    DATASET_DIR / "medredqa_val.csv",
    DATASET_DIR / "medredqa_test.csv",
]

assert DATASET_DIR.exists(), f"Missing dataset directory: {DATASET_DIR}"


def _record(question: str) -> Pass1Record:
    question = question.strip()
    return Pass1Record(id=sha256(question.encode("utf-8")).hexdigest(), question=question)


def _question_from_row(row: dict[str, str]) -> str:
    title = (row.get("Title") or "").strip()
    body = (row.get("Body") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def extract() -> Iterator[Pass1Record]:
    for path in tqdm(DATA_FILES, desc="medredqa files", unit="file"):
        assert path.is_file(), f"missing source file: {path}"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in tqdm(reader, desc=path.name, unit="row", leave=False):
                question = _question_from_row(row)
                if question:
                    yield _record(question)


def _count_records(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for row in csv.DictReader(handle) if _question_from_row(row))


def _dry_run() -> None:
    size_bytes = sum(path.stat().st_size for path in DATA_FILES if path.is_file())
    count = sum(_count_records(path) for path in DATA_FILES)
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
