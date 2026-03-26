"""Minimal Anthropic Bedrock CLI for one-off remote model checks.

Usage:
    uv run --env-file .env aws_utils.py --model anthropic-haiku-4.5
    uv run --env-file .env aws_utils.py --model anthropic-sonnet-4.6 --region us-west-2
    uv run --env-file .env aws_utils.py --model anthropic-opus-4.6 --prompt "who is the mayor of Dunedin, New Zealand"

This script sends exactly one request at a time. It is intentionally single-
request because this box is rate limited by Bedrock.
"""

import asyncio
import logging
import os
import time

import boto3
import click
from anthropic import AsyncAnthropicBedrock
from anthropic import BadRequestError
from anthropic import PermissionDeniedError
from anthropic import RateLimitError
from anthropic.types import Message

from cli_utils import CLICK_CONTEXT_SETTINGS
from utils import init_logging

_ = boto3
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "who is the mayor of Dunedin, New Zealand"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
AWS_BEDROCK_SUPPORTED_MODEL_IDS = {
    "anthropic-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "anthropic-opus-4.6": "us.anthropic.claude-opus-4-6-v1",
}
NOISY_LOGGERS = (
    "anthropic",
    "anthropic._base_client",
    "botocore",
    "botocore.auth",
    "botocore.credentials",
    "botocore.hooks",
    "botocore.session",
    "botocore.utils",
)


def resolve_bedrock_credentials() -> tuple[str | None, str | None, str | None]:
    access_key = os.environ.get("AWS_BEDROCK_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_BEDROCK_SESSION_TOKEN", "").strip() or None
    assert bool(access_key) == bool(secret_key), (
        "Set both AWS_BEDROCK_ACCESS_KEY_ID and AWS_BEDROCK_SECRET_ACCESS_KEY, or neither."
    )
    if access_key and secret_key:
        return access_key, secret_key, session_token
    return None, None, None


def extract_text_blocks(message: Message) -> str:
    visible_blocks: list[str] = []
    thinking_block_count = 0
    redacted_thinking_block_count = 0

    for block in message.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str) and text.strip():
                visible_blocks.append(text.strip())
            continue
        if block_type == "thinking":
            thinking_block_count += 1
            continue
        if block_type == "redacted_thinking":
            redacted_thinking_block_count += 1
            continue
        logger.warning("Ignoring unsupported Bedrock content block | type=%s", block_type)

    logger.info(
        "Parsed Bedrock response blocks | text_blocks=%d | thinking_blocks=%d | redacted_thinking_blocks=%d",
        len(visible_blocks),
        thinking_block_count,
        redacted_thinking_block_count,
    )
    visible_text = "\n\n".join(visible_blocks).strip()
    assert visible_text, "Bedrock response did not contain any visible text blocks."
    return visible_text


def resolve_region(region: str | None) -> str:
    resolved_region = (
        region
        or os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("AWS_DEFAULT_REGION", "").strip()
    )
    assert resolved_region, "Set AWS_REGION or pass --region."
    return resolved_region


async def ask_bedrock(
    *,
    model_name: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
    top_p: float | None = None,
) -> str:
    assert model_name in AWS_BEDROCK_SUPPORTED_MODEL_IDS, f"Unsupported Bedrock model: {model_name}"
    resolved_model_id = AWS_BEDROCK_SUPPORTED_MODEL_IDS[model_name]
    resolved_region = resolve_region(region.strip() if region is not None else None)
    normalized_prompt = prompt.strip()
    normalized_system_prompt = system_prompt.strip()
    if messages is None:
        assert normalized_prompt, (
            "prompt must be non-empty when messages are not provided."
        )
        request_messages = [{"role": "user", "content": normalized_prompt}]
    else:
        assert not normalized_prompt, (
            "prompt must be empty when messages are provided."
        )
        assert messages, "messages must be non-empty."
        request_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message["role"]).strip()
            content = str(message["content"]).strip()
            assert role in {"user", "assistant"}, (
                "Bedrock messages must use only user/assistant roles."
            )
            assert content, "Bedrock message content must be non-empty."
            request_messages.append({"role": role, "content": content})
    if top_p is not None:
        assert 0 < top_p <= 1, "top_p must be in (0, 1] when set."

    access_key, secret_key, session_token = resolve_bedrock_credentials()
    request_started_at = time.perf_counter()
    async with AsyncAnthropicBedrock(
        aws_access_key=access_key,
        aws_profile=profile,
        aws_region=resolved_region,
        aws_secret_key=secret_key,
        aws_session_token=session_token,
        max_retries=0,
        timeout=timeout_seconds,
    ) as client:
        logger.info(
            "Starting Bedrock request | model=%s | resolved_bedrock_model_id=%s | region=%s | credential_source=%s | max_tokens=%d | temperature=%.3f",
            model_name,
            resolved_model_id,
            resolved_region,
            "AWS_BEDROCK_*" if access_key is not None else "default AWS chain",
            max_tokens,
            temperature,
        )
        request_kwargs: dict[str, object] = {
            "model": resolved_model_id,
            "max_tokens": max_tokens,
            "messages": request_messages,
            "temperature": temperature,
        }
        if normalized_system_prompt:
            request_kwargs["system"] = normalized_system_prompt
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        response = await client.messages.create(**request_kwargs)
    request_duration_seconds = time.perf_counter() - request_started_at
    answer = extract_text_blocks(response)
    usage = response.usage
    logger.info(
        "Completed Bedrock request | model=%s | resolved_bedrock_model_id=%s | region=%s | stop_reason=%s | input_tokens=%s | output_tokens=%s | cache_creation_input_tokens=%s | cache_read_input_tokens=%s | duration_seconds=%.3f",
        model_name,
        response.model,
        resolved_region,
        response.stop_reason,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
        request_duration_seconds,
    )
    return answer


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS), case_sensitive=False),
    required=True,
    help="Anthropic shorthand to invoke on Bedrock.",
)
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_PROMPT,
    show_default=True,
    help="User prompt to send to the selected Bedrock model.",
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
    help="AWS profile override passed to the Anthropic Bedrock client when AWS_BEDROCK_* keys are not set.",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Maximum number of output tokens to request.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_TEMPERATURE,
    show_default=True,
    help="Sampling temperature for the request.",
)
@click.option(
    "--timeout-seconds",
    type=click.FloatRange(min=1.0),
    default=DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Request timeout in seconds.",
)
def cli(
    model_name: str,
    prompt: str,
    region: str | None,
    profile: str | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> None:
    """Send one prompt to Claude on Bedrock using a fixed model."""
    normalized_prompt = prompt.strip()
    assert normalized_prompt, "--prompt must be non-empty."
    normalized_profile = profile.strip() if profile is not None else None
    if normalized_profile is not None:
        assert normalized_profile, "--profile must be non-empty when set."
    resolved_region = resolve_region(region.strip() if region is not None else None)
    run_log_path = init_logging(level=logging.DEBUG)
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logger.info("Logging Bedrock CLI output to %s", run_log_path)
    try:
        answer = asyncio.run(
            ask_bedrock(
                model_name=model_name.lower(),
                region=resolved_region,
                profile=normalized_profile,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                prompt=normalized_prompt,
            )
        )
    except PermissionDeniedError as exc:
        raise click.ClickException(
            "Bedrock denied this request. Ensure the active AWS identity has "
            "model access enabled and an IAM policy that allows "
            "`bedrock:InvokeModel` for the selected model."
        ) from exc
    except BadRequestError as exc:
        raise click.ClickException(str(exc)) from exc
    except RateLimitError as exc:
        raise click.ClickException(
            "Bedrock rate limited the request. This CLI only sends one request "
            "at a time, so retry later or lower your overall account traffic."
        ) from exc
    click.echo(answer)


if __name__ == "__main__":
    cli()
