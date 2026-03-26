"""Generate pass-2 JSONL records from a pass-1 JSONL input file.

Usage:
    uv run --env-file .env -m main --help
    uv run --env-file .env -m main
    uv run --env-file .env -m main --output-path all_consumer_health_questions.pass2.jsonl

This script is intentionally linear and single-request-at-a-time because the
Bedrock client in this project runs with retries disabled and the box is rate
limited. It validates every input line as a Pass1Record, generates a matching
Pass2Record, and appends one JSON object per line to the output file.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from aws_utils import AWS_BEDROCK_SUPPORTED_MODEL_IDS
from entities import Pass1Record
from generator import generate_pass2_record

logger = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = Path("all_consumer_health_questions.jsonl")
DEFAULT_MODEL_NAME = "anthropic-haiku-4.5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pass-2 JSONL file from pass-1 consumer health questions."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input pass-1 JSONL file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output pass-2 JSONL file. Defaults to <input stem>.pass2.jsonl.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS),
        default=DEFAULT_MODEL_NAME,
        help="Bedrock model shorthand defined in aws_utils.py.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region override. Defaults to AWS_REGION or AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile override passed through to generator.generate_pass2_record().",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of output tokens to request. Defaults to {DEFAULT_MAX_TOKENS}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for the request. Defaults to {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds. Defaults to {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert args.input_path.is_file(), f"Input file not found: {args.input_path}"
    assert args.max_tokens >= 1, "--max-tokens must be at least 1."
    assert 0.0 <= args.temperature <= 1.0, "--temperature must be between 0.0 and 1.0."
    assert args.timeout_seconds >= 1.0, "--timeout-seconds must be at least 1.0."

    output_path = args.output_path
    if output_path is None:
        suffixless_name = (
            args.input_path.name.removesuffix(".jsonl")
            if args.input_path.name.endswith(".jsonl")
            else args.input_path.name
        )
        output_path = args.input_path.with_name(f"{suffixless_name}.pass2.jsonl")
    assert output_path != args.input_path, "--output-path must differ from --input-path."
    if output_path.exists():
        assert args.overwrite, (
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    with args.input_path.open("r", encoding="utf-8") as input_file:
        total_records = sum(1 for _ in input_file)
    assert total_records > 0, f"Input file is empty: {args.input_path}"

    started_at = time.perf_counter()
    logger.info(
        "Starting pass-2 generation | input=%s | output=%s | total_records=%d | model=%s",
        args.input_path,
        output_path,
        total_records,
        args.model_name,
    )

    with logging_redirect_tqdm():
        with (
            args.input_path.open("r", encoding="utf-8") as input_file,
            output_path.open("w", encoding="utf-8") as output_file,
            tqdm(
                total=total_records,
                desc="pass2",
                unit="record",
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as progress,
        ):
            for line_number, line in enumerate(input_file, start=1):
                stripped_line = line.strip()
                assert stripped_line, f"Blank JSONL line at input line {line_number}."
                try:
                    pass1_record = Pass1Record.model_validate_json(stripped_line)
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid Pass1Record at input line {line_number}: {exc}\n{stripped_line}"
                    ) from exc
                logger.info(
                    "Generating Pass2Record | line=%d | id=%s",
                    line_number,
                    pass1_record.id,
                )
                try:
                    pass2_record = await generate_pass2_record(
                        pass1_record,
                        model_name=args.model_name,
                        region=args.region,
                        profile=args.profile,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_seconds=args.timeout_seconds,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to generate Pass2Record at input line {line_number} for id={pass1_record.id}."
                    ) from exc
                output_file.write(pass2_record.model_dump_json())
                output_file.write("\n")
                output_file.flush()
                logger.info(
                    "Wrote Pass2Record | line=%d | id=%s | output=%s",
                    line_number,
                    pass2_record.id,
                    output_path,
                )
                progress.update(1)

    logger.info(
        "Completed pass-2 generation | input=%s | output=%s | total_records=%d | duration_seconds=%.3f",
        args.input_path,
        output_path,
        total_records,
        time.perf_counter() - started_at,
    )


if __name__ == "__main__":
    asyncio.run(main())
