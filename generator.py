from __future__ import annotations

import argparse
import asyncio

from pydantic import ValidationError

from aws_utils import AWS_BEDROCK_SUPPORTED_MODEL_IDS
from aws_utils import ask_bedrock
from entities import Pass1Record
from entities import Pass2Record
from samples import PASS1_RECORD_EXAMPLE_JSON

SYSTEM_PROMPT = """
You are generating structured annotations for a consumer health question dataset.
Return exactly one JSON object and nothing else.
Do not wrap the JSON in markdown fences.
Keep the same id and question from the provided input record.
""".strip()


async def generate_pass2_record(
    pass1_record: Pass1Record,
    *,
    model_name: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> Pass2Record:
    response_text = await ask_bedrock(
        model_name=model_name,
        region=region,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            "Convert this Pass1Record into a Pass2Record.\n"
            "Requirements:\n"
            "- Keep id exactly unchanged.\n"
            "- Keep question exactly unchanged.\n"
            "- Set difficulty_level to an integer from 1 to 5.\n"
            "- Produce 3 to 7 follow-up questions.\n"
            "- Each follow-up must contain question, thinking, and weight.\n"
            "- Each weight must be an integer from 1 to 5.\n"
            "- Follow-up questions must be distinct after lowercasing and collapsing whitespace.\n"
            "- Return JSON only.\n\n"
            f"Pass1Record JSON:\n{pass1_record.model_dump_json()}"
        ),
    )
    try:
        pass2_record = Pass2Record.model_validate_json(response_text)
    except ValidationError as exc:
        raise ValueError(f"Bedrock response did not parse as Pass2Record:\n{response_text}") from exc
    assert pass2_record.id == pass1_record.id, "Pass2Record.id must match Pass1Record.id."
    assert pass2_record.question == pass1_record.question, (
        "Pass2Record.question must match Pass1Record.question."
    )
    return pass2_record


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Pass2Record from a Pass1Record with Claude Haiku on Bedrock."
    )
    parser.add_argument(
        "--pass1-json",
        default=PASS1_RECORD_EXAMPLE_JSON,
        help="Pass1Record JSON. Defaults to the sample payload from samples.py.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS),
        default="anthropic-haiku-4.5",
        help="Bedrock model shorthand defined in aws_utils.py.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region override. Defaults to AWS_REGION or AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile override passed through to aws_utils.ask_bedrock().",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of output tokens to request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the request.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )
    args = parser.parse_args()
    pass1_record = Pass1Record.model_validate_json(args.pass1_json)
    pass2_record = await generate_pass2_record(
        pass1_record,
        model_name=args.model_name,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    print(pass2_record.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
