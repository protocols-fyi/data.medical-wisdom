from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import click
from pydantic import ValidationError

from aws_utils import AWS_BEDROCK_SUPPORTED_MODEL_IDS
from aws_utils import BedrockResponse
from aws_utils import NOISY_LOGGERS
from aws_utils import ask_bedrock
from entities import Pass1Record
from entities import Pass2Record
from samples import PASS1_RECORD_EXAMPLE_JSON

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """
You generate high-information-gain clarification questions for consumer health enquiries.
Be clinically cautious, prioritize the missing context that most changes the answer, and follow the provided JSON schema exactly.
""".strip()


def sanitize_json_schema(node: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in node:
        reference = node["$ref"]
        assert reference.startswith("#/$defs/"), f"Unsupported schema reference: {reference}"
        definition_name = reference.removeprefix("#/$defs/")
        return sanitize_json_schema(definitions[definition_name], definitions)

    node_type = node.get("type")
    if node_type == "object":
        return {
            "type": "object",
            "properties": {
                key: sanitize_json_schema(value, definitions)
                for key, value in node["properties"].items()
            },
            "required": list(node.get("required", [])),
            "additionalProperties": bool(node.get("additionalProperties", False)),
        }
    if node_type == "array":
        return {
            "type": "array",
            "items": sanitize_json_schema(node["items"], definitions),
        }
    if node_type == "integer":
        minimum = node.get("minimum")
        maximum = node.get("maximum")
        if minimum == 1 and maximum == 5:
            return {"type": "integer", "enum": [1, 2, 3, 4, 5]}
        return {"type": "integer"}
    if node_type == "string":
        if "const" in node:
            return {"type": "string", "enum": [node["const"]]}
        return {"type": "string"}
    assert "const" not in node, f"Unsupported schema node without type: {node}"
    return {"type": node_type}


def build_pass2_output_schema(pass1_record: Pass1Record) -> dict[str, Any]:
    # Bedrock structured output needs a schema specialized to this Pass1Record:
    # start from the Pydantic Pass2Record schema, reduce it to Anthropic's
    # supported JSON Schema subset, then remove id/question so the model does
    # not have to re-emit long source text with embedded newlines or quotes.
    _ = pass1_record
    generated_schema = Pass2Record.model_json_schema()
    definitions = generated_schema.get("$defs", {})
    schema = sanitize_json_schema(generated_schema, definitions)
    del schema["properties"]["id"]
    del schema["properties"]["question"]
    schema["required"] = [
        field_name
        for field_name in schema["required"]
        if field_name not in {"id", "question"}
    ]
    return schema


def build_generation_prompt(pass1_record: Pass1Record) -> str:
    return (
        "Your job is to carefully consider the question asked in Pass1Record, "
        "ultrathink about what are the highest information-gain follow-up "
        "questions you can ask to significantly reduce ambiguity, improve "
        "context and situational awareness. You should also provide concise "
        "explanations on why you want to ask each follow-up question and what "
        "the answer mean.\n"
        "Do not repeat the id or question fields; they will be filled in by the caller.\n"
        "Return 3 to 7 follow-up questions, ordered from highest to lowest information gain.\n"
        "Prefer specific missing context over generic filler, and keep the "
        "record compact while still telling the full story of why each "
        "follow-up matters.\n\n"
        f"Pass1Record JSON:\n{pass1_record.model_dump_json()}"
    )


async def generate_pass2_record(
    pass1_record: Pass1Record,
    *,
    model_name: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> tuple[Pass2Record, BedrockResponse]:
    output_schema = build_pass2_output_schema(pass1_record)
    prompt = build_generation_prompt(pass1_record)
    attempt_max_tokens = max_tokens
    bedrock_response: BedrockResponse | None = None

    for attempt_number in (1, 2):
        bedrock_response = await ask_bedrock(
            model_name=model_name,
            region=region,
            profile=profile,
            max_tokens=attempt_max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            system_prompt=SYSTEM_PROMPT,
            json_output_schema=output_schema,
            prompt=prompt,
            log_response_summary=False,
        )
        try:
            pass2_payload = json.loads(bedrock_response.text, strict=False)
            pass2_payload["id"] = pass1_record.id
            pass2_payload["question"] = pass1_record.question
            if (
                isinstance(pass2_payload.get("follow_ups"), list)
                and len(pass2_payload["follow_ups"]) > 7
            ):
                logger.warning(
                    "Trimming excess follow_ups before validation | id=%s | original_count=%d",
                    pass1_record.id,
                    len(pass2_payload["follow_ups"]),
                )
                pass2_payload["follow_ups"] = pass2_payload["follow_ups"][:7]
            pass2_record = Pass2Record.model_validate(pass2_payload)
        except (ValidationError, json.JSONDecodeError) as exc:
            if attempt_number == 1:
                retry_max_tokens = max(attempt_max_tokens * 2, 2048)
                logger.warning(
                    "Retrying Pass2Record generation after malformed model output | id=%s | attempt=%d | max_tokens=%d | retry_max_tokens=%d | error=%s",
                    pass1_record.id,
                    attempt_number,
                    attempt_max_tokens,
                    retry_max_tokens,
                    exc,
                )
                attempt_max_tokens = retry_max_tokens
                continue
            raise ValueError(
                "Bedrock response did not parse as Pass2Record:\n"
                f"{bedrock_response.text}\n\n"
                f"Schema used:\n{json.dumps(output_schema, indent=2)}"
            ) from exc
        assert pass2_record.id == pass1_record.id, "Pass2Record.id must match Pass1Record.id."
        assert pass2_record.question == pass1_record.question, (
            "Pass2Record.question must match Pass1Record.question."
        )
        return pass2_record, bedrock_response

    raise AssertionError("Pass2 generation retry loop must return or raise.")


async def main(
    *,
    pass1_json: str,
    model_name: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    pass1_record = Pass1Record.model_validate_json(pass1_json)
    pass2_record, _ = await generate_pass2_record(
        pass1_record,
        model_name=model_name,
        region=region,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
    click.echo(pass2_record.model_dump_json(indent=2))


@click.command()
@click.option(
    "--pass1-json",
    type=str,
    default=PASS1_RECORD_EXAMPLE_JSON,
    show_default=True,
    help="Pass1Record JSON. Defaults to the sample payload from samples.py.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS), case_sensitive=False),
    default="anthropic-haiku-4.5",
    show_default=True,
    help="Bedrock model shorthand defined in aws_utils.py.",
)
@click.option(
    "--region",
    type=str,
    default=None,
    help="AWS region override. Defaults to AWS_REGION or AWS_DEFAULT_REGION.",
)
@click.option(
    "--profile",
    type=str,
    default=None,
    help="AWS profile override passed through to aws_utils.ask_bedrock().",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
    help="Maximum number of output tokens to request.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0,
    show_default=True,
    help="Sampling temperature for the request.",
)
@click.option(
    "--timeout-seconds",
    type=click.FloatRange(min=1.0),
    default=120.0,
    show_default=True,
    help="Request timeout in seconds.",
)
def cli(
    pass1_json: str,
    model_name: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> None:
    """Generate a Pass2Record from a Pass1Record with Claude Haiku on Bedrock."""
    asyncio.run(
        main(
            pass1_json=pass1_json,
            model_name=model_name.lower(),
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    )


if __name__ == "__main__":
    cli()
