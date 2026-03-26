from __future__ import annotations

from collections.abc import Iterator
import xml.etree.ElementTree as ET
from hashlib import sha256
from itertools import islice
from pathlib import Path

import click
from tqdm import tqdm

from entities import Pass1Record

DATASET_DIR = Path("datasets.original/mediqa_qa/MEDIQA2019-master/MEDIQA_Task3_QA")
SOURCE_FILES = sorted(DATASET_DIR.glob("*.xml"))

assert DATASET_DIR.exists(), f"Missing dataset directory: {DATASET_DIR}"


def _question_texts() -> list[str]:
    texts: list[str] = []
    for path in SOURCE_FILES:
        root = ET.parse(path).getroot()
        for question in root.findall("Question"):
            text = question.findtext("QuestionText")
            if text:
                texts.append(text.strip())
    return texts


def _record(question: str) -> Pass1Record:
    return Pass1Record.model_validate(
        {
            "id": sha256(question.encode("utf-8")).hexdigest(),
            "question": question,
        }
    )


def extract() -> Iterator[Pass1Record]:
    for question in _question_texts():
        yield _record(question)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print dataset location and size/count.")
@click.option("--limit", type=int, default=None, help="Only emit the first N parsed records.")
def cli(dry_run: bool, limit: int | None) -> None:
    questions = _question_texts()
    if dry_run:
        size = sum(path.stat().st_size for path in SOURCE_FILES)
        click.echo(f"location: {DATASET_DIR}")
        click.echo(f"files: {len(SOURCE_FILES)}")
        click.echo(f"size_bytes: {size}")
        click.echo(f"questions: {len(questions)}")
        return

    selected = islice(questions, limit) if limit is not None else questions
    total = len(questions) if limit is None else min(limit, len(questions))
    for record in tqdm(selected, total=total, desc="mediqa_qa", unit="record"):
        click.echo(_record(record).model_dump_json())


if __name__ == "__main__":
    cli()
