"""File IO for pass-1 inputs and pass-2 artifacts.

Inputs:
- Paths and JSON-serializable rows.

Outputs:
- Loaded pass-1 records.
- JSONL and JSON artifact files written to disk.
"""

import json
from pathlib import Path


def load_pass1_records(path: Path, *, limit: int) -> list[dict[str, str]]:
    resolved_path = path.expanduser().resolve()
    assert resolved_path.is_file(), f"Pass-1 JSONL not found: {resolved_path}"
    records: list[dict[str, str]] = []
    with resolved_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not (line := raw_line.strip()):
                continue
            payload = json.loads(line)
            assert set(payload) == {"dataset", "id", "question"}, (
                f"Unexpected pass-1 fields in {resolved_path}: {sorted(payload)}"
            )
            normalized_record = {
                "dataset": str(payload["dataset"]).strip(),
                "id": str(payload["id"]).strip(),
                "question": str(payload["question"]).strip(),
            }
            assert all(normalized_record.values()), (
                f"Pass-1 record has empty required fields: {payload}"
            )
            records.append(normalized_record)
            if limit > 0 and len(records) >= limit:
                break
    assert records, f"No pass-1 records found in {resolved_path}."
    return records


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def append_jsonl_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def clear_output_files(paths: list[Path]) -> None:
    for path in paths:
        resolved_path = path.expanduser().resolve()
        if resolved_path.exists():
            resolved_path.unlink()
