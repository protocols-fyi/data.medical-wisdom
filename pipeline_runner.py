"""Sequential pass-2 execution through existing Bedrock calls.

Inputs:
- RuntimeConfig, pass-1 records, and output artifact paths.

Outputs:
- Valid enriched rows appended to JSONL.
- Repair attempts and permanent failures written to JSONL sidecars.
- A run manifest describing processed, skipped, and failed rows.
"""

import asyncio
from datetime import UTC, datetime
import logging
from pathlib import Path

from aws_utils import ask_bedrock

from pipeline_config import (
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_REPAIR_BACKOFF_SECONDS,
    RuntimeConfig,
)
from pipeline_io import append_jsonl_row, clear_output_files, write_json
from pipeline_validation import (
    index_source_records,
    load_existing_valid_keys,
    logical_key_for_record,
    parse_annotation_response,
)

logger = logging.getLogger(__name__)

ANNOTATION_SYSTEM_PROMPT = """You are annotating consumer-health questions.

Return JSON only with this exact shape:
{
  "id": "string",
  "dataset": "string",
  "user": "string",
  "difficulty_level": 1,
  "questions": [
    {
      "question": "string",
      "thinking": "string",
      "weight": 1
    }
  ]
}

Rules:
- Preserve dataset, id, and user exactly as provided.
- Generate 3 to 7 follow-up questions.
- Make the follow-up questions non-redundant and target distinct information gaps.
- Make weight and difficulty_level integers from 1 to 5.
- Do not wrap the JSON in triple backticks or ```json fences.
- Avoid markdown, code fences, and extra commentary.
"""


def build_annotation_user_prompt(record: dict[str, str]) -> str:
    return "\n".join(
        [
            f"dataset: {record['dataset']}",
            f"id: {record['id']}",
            f"question: {record['question']}",
            "",
            "Return one JSON object only.",
            "The top-level fields must be id, dataset, user, difficulty_level, questions.",
            "The user field must exactly equal the original question.",
            "Each follow-up entry must include question, thinking, and weight.",
            "Do not wrap the response in markdown fences.",
        ]
    )


def build_repair_user_prompt(
    record: dict[str, str],
    invalid_output: str,
    validation_errors: list[str],
) -> str:
    error_lines = [f"- {error}" for error in validation_errors]
    return "\n".join(
        [
            "The previous response was invalid. Repair it.",
            "",
            "Original input:",
            f"dataset: {record['dataset']}",
            f"id: {record['id']}",
            f"question: {record['question']}",
            "",
            "Invalid output:",
            invalid_output.strip(),
            "",
            "Validation errors:",
            *error_lines,
            "",
            "Return corrected JSON only.",
            "Preserve dataset, id, and user exactly.",
            "Generate 3 to 7 non-redundant follow-up questions.",
            "Do not wrap the response in markdown fences.",
        ]
    )


async def request_annotation(
    cfg: RuntimeConfig,
    record: dict[str, str],
    *,
    invalid_output: str | None = None,
    validation_errors: list[str] | None = None,
) -> str:
    prompt = (
        build_annotation_user_prompt(record)
        if invalid_output is None
        else build_repair_user_prompt(record, invalid_output, validation_errors or [])
    )
    return await ask_bedrock(
        model_name=cfg.model_name,
        region=cfg.region,
        profile=cfg.profile,
        max_tokens=cfg.max_tokens,
        temperature=0.0,
        timeout_seconds=cfg.timeout_seconds,
        prompt=prompt,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
    )


async def annotate_record(
    cfg: RuntimeConfig,
    record: dict[str, str],
) -> dict[str, object]:
    repair_attempt_logs: list[dict[str, object]] = []
    last_raw_text = await request_annotation(cfg, record)
    payload, validation_errors = parse_annotation_response(
        source_record=record,
        response_text=last_raw_text,
    )
    if payload is not None:
        return {
            "ok": True,
            "payload": payload,
            "repairAttempts": repair_attempt_logs,
            "attemptCount": 1,
        }

    for repair_attempt_number, backoff_seconds in enumerate(
        DEFAULT_REPAIR_BACKOFF_SECONDS[:DEFAULT_MAX_REPAIR_ATTEMPTS],
        start=1,
    ):
        logger.info(
            "Repairing invalid annotation | logical_key=%s | repair_attempt=%d | sleep_seconds=%.1f | errors=%s",
            logical_key_for_record(record),
            repair_attempt_number,
            backoff_seconds,
            validation_errors,
        )
        await asyncio.sleep(backoff_seconds)
        repaired_raw_text = await request_annotation(
            cfg,
            record,
            invalid_output=last_raw_text,
            validation_errors=validation_errors,
        )
        repaired_payload, repaired_validation_errors = parse_annotation_response(
            source_record=record,
            response_text=repaired_raw_text,
        )
        repair_attempt_logs.append(
            {
                "logicalKey": logical_key_for_record(record),
                "repairAttempt": repair_attempt_number,
                "priorInvalidOutput": last_raw_text,
                "priorErrors": validation_errors,
                "repairedOutput": repaired_raw_text,
                "repairedOutputErrors": repaired_validation_errors,
            }
        )
        last_raw_text = repaired_raw_text
        validation_errors = repaired_validation_errors
        if repaired_payload is not None:
            return {
                "ok": True,
                "payload": repaired_payload,
                "repairAttempts": repair_attempt_logs,
                "attemptCount": 1 + repair_attempt_number,
            }

    return {
        "ok": False,
        "repairAttempts": repair_attempt_logs,
        "attemptCount": 1 + len(repair_attempt_logs),
        "errors": validation_errors,
        "lastRawText": last_raw_text,
    }


async def annotate_records(
    cfg: RuntimeConfig,
    *,
    records: list[dict[str, str]],
    output_jsonl: Path,
    failures_jsonl: Path,
    repair_attempts_jsonl: Path,
    run_manifest_json: Path,
    resume: bool,
) -> dict[str, object]:
    source_records_by_key = index_source_records(records)
    resolved_output_jsonl = output_jsonl.expanduser().resolve()
    resolved_failures_jsonl = failures_jsonl.expanduser().resolve()
    resolved_repair_attempts_jsonl = repair_attempts_jsonl.expanduser().resolve()
    resolved_run_manifest_json = run_manifest_json.expanduser().resolve()
    existing_valid_keys = (
        load_existing_valid_keys(resolved_output_jsonl, source_records_by_key)
        if resume
        else set()
    )
    if not resume:
        clear_output_files(
            [
                resolved_output_jsonl,
                resolved_failures_jsonl,
                resolved_repair_attempts_jsonl,
                resolved_run_manifest_json,
            ]
        )

    logger.info(
        "Starting pass-2 annotation run | records=%d | resume=%s | existing_valid=%d | output=%s",
        len(records),
        resume,
        len(existing_valid_keys),
        resolved_output_jsonl,
    )
    started_at_utc = datetime.now(UTC).isoformat()
    success_count = 0
    failure_count = 0
    skipped_count = 0
    processed_keys: list[str] = []
    for index, record in enumerate(records, start=1):
        logical_key = logical_key_for_record(record)
        if logical_key in existing_valid_keys:
            skipped_count += 1
            logger.info(
                "Skipping already-valid enriched row | index=%d | logical_key=%s",
                index,
                logical_key,
            )
            continue
        logger.info(
            "Annotating record | index=%d | logical_key=%s",
            index,
            logical_key,
        )
        result = await annotate_record(cfg, record)
        processed_keys.append(logical_key)
        for repair_attempt in result["repairAttempts"]:
            append_jsonl_row(resolved_repair_attempts_jsonl, repair_attempt)
        if bool(result["ok"]):
            append_jsonl_row(resolved_output_jsonl, result["payload"])
            success_count += 1
            continue
        append_jsonl_row(
            resolved_failures_jsonl,
            {
                "dataset": record["dataset"],
                "id": record["id"],
                "question": record["question"],
                "logicalKey": logical_key,
                "attemptCount": result["attemptCount"],
                "errors": result["errors"],
                "lastRawText": result["lastRawText"],
            },
        )
        failure_count += 1

    manifest = {
        "startedAtUtc": started_at_utc,
        "completedAtUtc": datetime.now(UTC).isoformat(),
        "modelName": cfg.model_name,
        "region": cfg.region,
        "inputRecordCount": len(records),
        "processedRecordCount": len(processed_keys),
        "successCount": success_count,
        "failureCount": failure_count,
        "skippedCount": skipped_count,
        "resume": resume,
        "limit": cfg.limit,
        "outputJsonl": str(resolved_output_jsonl),
        "failuresJsonl": str(resolved_failures_jsonl),
        "repairAttemptsJsonl": str(resolved_repair_attempts_jsonl),
        "processedLogicalKeys": processed_keys,
    }
    write_json(resolved_run_manifest_json, manifest)
    logger.info(
        "Completed pass-2 annotation run | success=%d | failure=%d | skipped=%d | manifest=%s",
        success_count,
        failure_count,
        skipped_count,
        resolved_run_manifest_json,
    )
    return manifest
