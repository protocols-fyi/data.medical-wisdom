from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import click
from tqdm import tqdm

from entities import Pass1Record

DATASET_FILE = Path(
    "datasets.original/liveqa_med/"
    "LiveQA_MedicalTask_TREC2017-master/TestDataset/"
    "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
)

assert DATASET_FILE.exists(), f"Missing dataset file: {DATASET_FILE}"


def _question_texts() -> Iterator[str]:
    root = ET.parse(DATASET_FILE).getroot()
    for node in root.iterfind(".//NLM-QUESTION"):
        subject = (node.findtext("./Original-Question/SUBJECT") or "").strip()
        message = (node.findtext("./Original-Question/MESSAGE") or "").strip()
        summary = (node.findtext("./NLM-Summary") or "").strip()
        if subject and message:
            text = f"{subject}\n\n{message}"
        else:
            text = subject or message or summary
        text = text.strip()
        if text:
            yield text


def _record_from_question(question: str) -> Pass1Record:
    return Pass1Record(id=sha256(question.encode("utf-8")).hexdigest(), question=question)


def _count_questions() -> int:
    return sum(1 for _ in _question_texts())


def _source_size_bytes() -> int:
    return DATASET_FILE.stat().st_size


def extract() -> Iterator[Pass1Record]:
    for question in _question_texts():
        yield _record_from_question(question)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print dataset location and size/count, then exit.")
@click.option("--limit", type=int, default=None, help="Limit emitted records to the first N items.")
def cli(dry_run: bool, limit: int | None) -> None:
    if limit is not None and limit < 1:
        raise click.BadParameter("--limit must be at least 1")

    record_count = _count_questions()
    if dry_run:
        click.echo(f"location={DATASET_FILE} size_bytes={_source_size_bytes()} records={record_count}")
        return

    for index, record in enumerate(tqdm(extract(), total=record_count, unit="record"), start=1):
        click.echo(record.model_dump_json())
        if limit is not None and index >= limit:
            break


if __name__ == "__main__":
    cli()
