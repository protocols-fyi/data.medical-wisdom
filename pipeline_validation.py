"""Validation and resume checks for enriched outputs.

Inputs:
- Pass-1 source records and raw model responses or enriched JSONL paths.

Outputs:
- Stable logical keys.
- Validation error lists or parsed payloads.
- Resume-safe sets of already-valid rows.
- Final validation summaries for enriched JSONL files.
"""

from difflib import SequenceMatcher
import json
from pathlib import Path
import re

SEMANTIC_AXES = {
    "urgency": (
        "urgent",
        "emergency",
        "red flag",
        "chest pain",
        "shortness of breath",
        "faint",
        "fainting",
        "trouble speaking",
        "severe",
        "danger",
    ),
    "chronology": (
        "when",
        "how long",
        "how often",
        "started",
        "worse",
        "better",
        "sudden",
        "severity",
        "frequency",
    ),
    "patient_context": (
        "you or someone else",
        "age",
        "pregnant",
        "child",
        "adult",
        "who is this about",
    ),
    "prior_evaluation": (
        "diagnosed",
        "tested",
        "evaluation",
        "doctor",
        "exam",
        "scan",
        "lab",
        "already seen",
    ),
    "history_and_meds": (
        "medication",
        "medicine",
        "blood thinner",
        "history",
        "condition",
        "allergy",
        "other health conditions",
    ),
    "intent": (
        "what are you most worried about",
        "what are you hoping to learn",
        "goal",
        "what do you want to know",
        "main concern",
    ),
}


def logical_key_for_record(record: dict[str, str]) -> str:
    return f"{record['dataset']}:{record['id']}"


def index_source_records(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    records_by_key: dict[str, dict[str, str]] = {}
    for record in records:
        logical_key = logical_key_for_record(record)
        assert logical_key not in records_by_key, f"Duplicate pass-1 logical key: {logical_key}"
        records_by_key[logical_key] = record
    return records_by_key


def normalize_text_for_semantics(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def classify_information_axes(text: str) -> set[str]:
    normalized_text = normalize_text_for_semantics(text)
    return {
        axis
        for axis, keywords in SEMANTIC_AXES.items()
        if any(keyword in normalized_text for keyword in keywords)
    }


def validate_annotation_payload(
    *,
    source_record: dict[str, str],
    payload: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    if payload.get("id") != source_record["id"]:
        errors.append("id did not round-trip exactly.")
    if payload.get("dataset") != source_record["dataset"]:
        errors.append("dataset did not round-trip exactly.")
    if payload.get("user") != source_record["question"]:
        errors.append("user did not round-trip exactly.")

    difficulty_level = payload.get("difficulty_level")
    if not isinstance(difficulty_level, int) or not (1 <= difficulty_level <= 5):
        errors.append("difficulty_level must be an integer in [1, 5].")

    questions = payload.get("questions")
    if not isinstance(questions, list) or not (3 <= len(questions) <= 7):
        errors.append("questions must be an array of length 3 to 7.")
        return errors

    normalized_user = normalize_text_for_semantics(source_record["question"])
    normalized_questions: list[str] = []
    all_axes: set[str] = set()
    for index, followup in enumerate(questions, start=1):
        if not isinstance(followup, dict):
            errors.append(f"questions[{index}] must be an object.")
            continue
        question = str(followup.get("question", "")).strip()
        thinking = str(followup.get("thinking", "")).strip()
        weight = followup.get("weight")
        if not question:
            errors.append(f"questions[{index}].question must be non-empty.")
        if not thinking:
            errors.append(f"questions[{index}].thinking must be non-empty.")
        if not isinstance(weight, int) or not (1 <= weight <= 5):
            errors.append(f"questions[{index}].weight must be an integer in [1, 5].")
        normalized_question = normalize_text_for_semantics(question)
        normalized_questions.append(normalized_question)
        if normalized_question == normalized_user:
            errors.append(f"questions[{index}] is a trivial restatement of the user question.")
        if len(thinking) < 40:
            errors.append(
                f"questions[{index}].thinking is too short to explain information gain clearly."
            )
        all_axes.update(classify_information_axes(question))

    if len(set(normalized_questions)) != len(normalized_questions):
        errors.append("Follow-up questions are not unique after normalization.")

    for left_index in range(len(normalized_questions)):
        for right_index in range(left_index + 1, len(normalized_questions)):
            similarity = SequenceMatcher(
                None,
                normalized_questions[left_index],
                normalized_questions[right_index],
            ).ratio()
            if similarity >= 0.88:
                errors.append(
                    f"questions[{left_index + 1}] and questions[{right_index + 1}] are near-duplicates (similarity={similarity:.3f})."
                )

    if len(all_axes) < 3:
        errors.append("Follow-up set did not cover at least 3 distinct information axes.")
    return errors


def parse_annotation_response(
    *,
    source_record: dict[str, str],
    response_text: str,
) -> tuple[dict[str, object] | None, list[str]]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return None, [f"Model output was not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["Model output JSON must be an object."]
    validation_errors = validate_annotation_payload(
        source_record=source_record,
        payload=payload,
    )
    if validation_errors:
        return None, validation_errors
    return payload, []


def load_existing_valid_keys(
    output_jsonl: Path,
    source_records_by_key: dict[str, dict[str, str]],
) -> set[str]:
    resolved_output_jsonl = output_jsonl.expanduser().resolve()
    if not resolved_output_jsonl.exists():
        return set()
    valid_keys: set[str] = set()
    invalid_rows: list[str] = []
    with resolved_output_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not (line := raw_line.strip()):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_rows.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            logical_key = f"{payload.get('dataset')}:{payload.get('id')}"
            source_record = source_records_by_key.get(logical_key)
            if source_record is None:
                invalid_rows.append(f"line {line_number}: unknown logical key {logical_key}")
                continue
            validation_errors = validate_annotation_payload(
                source_record=source_record,
                payload=payload,
            )
            if validation_errors:
                invalid_rows.append(
                    f"line {line_number}: {logical_key} failed validation ({'; '.join(validation_errors)})"
                )
                continue
            if logical_key in valid_keys:
                invalid_rows.append(f"line {line_number}: duplicate logical key {logical_key}")
                continue
            valid_keys.add(logical_key)
    assert not invalid_rows, (
        f"Existing output JSONL is not clean enough to resume: {resolved_output_jsonl}\n"
        + "\n".join(invalid_rows[:10])
    )
    return valid_keys


def validate_enriched_output(
    *,
    source_records: list[dict[str, str]],
    output_jsonl: Path,
    limit: int,
) -> dict[str, object]:
    source_records_by_key = index_source_records(source_records)
    resolved_output_jsonl = output_jsonl.expanduser().resolve()
    assert resolved_output_jsonl.is_file(), f"Enriched JSONL not found: {resolved_output_jsonl}"
    seen_output_keys: set[str] = set()
    validation_errors: list[dict[str, object]] = []
    success_count = 0
    processed_count = 0
    with resolved_output_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not (line := raw_line.strip()):
                continue
            payload = json.loads(line)
            logical_key = f"{payload.get('dataset')}:{payload.get('id')}"
            source_record = source_records_by_key.get(logical_key)
            if source_record is None:
                validation_errors.append(
                    {
                        "line": line_number,
                        "logicalKey": logical_key,
                        "errors": [f"Unknown logical key {logical_key}"],
                    }
                )
                continue
            if logical_key in seen_output_keys:
                validation_errors.append(
                    {
                        "line": line_number,
                        "logicalKey": logical_key,
                        "errors": [f"Duplicate output logical key {logical_key}"],
                    }
                )
                continue
            row_errors = validate_annotation_payload(
                source_record=source_record,
                payload=payload,
            )
            if row_errors:
                validation_errors.append(
                    {
                        "line": line_number,
                        "logicalKey": logical_key,
                        "errors": row_errors,
                    }
                )
                continue
            seen_output_keys.add(logical_key)
            success_count += 1
            processed_count += 1
            if limit > 0 and processed_count >= limit:
                break
    return {
        "outputJsonl": str(resolved_output_jsonl),
        "validatedCount": success_count,
        "errorCount": len(validation_errors),
        "errors": validation_errors,
    }
