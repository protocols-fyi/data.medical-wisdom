"""CLI entrypoint for the sequential Bedrock enrichment pipeline.

Inputs:
- CLI flags and command-specific file paths.

Outputs:
- Dry-run previews for every command.
- Written pass-1 smoke files, enriched JSONL outputs, validation summaries,
  and run artifacts through the helper modules.
"""

import asyncio
import json
import logging
from pathlib import Path

import click

from aws_utils import AWS_BEDROCK_SUPPORTED_MODEL_IDS, NOISY_LOGGERS
from cli_utils import CLICK_CONTEXT_SETTINGS
from pipeline_config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    RuntimeConfig,
    SMOKE_RECORDS,
)
from pipeline_io import load_pass1_records, write_json, write_jsonl
from pipeline_runner import annotate_records
from pipeline_validation import validate_enriched_output
from utils import init_logging

logger = logging.getLogger(__name__)


@click.group(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Log what would happen without making remote model calls or writing output files.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Maximum number of logical records to process. Use 0 for all available records.",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    show_default=True,
    help="AWS region for Bedrock runtime calls.",
)
@click.option(
    "--profile",
    type=str,
    default=None,
    help="Optional AWS profile name for Bedrock runtime calls.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS), case_sensitive=False),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Claude shorthand to invoke on Bedrock.",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Maximum completion tokens for each annotation request.",
)
@click.option(
    "--timeout-seconds",
    type=click.FloatRange(min=1.0),
    default=DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--workdir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("run"),
    show_default=True,
    help="Local working directory for generated artifacts and logs.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    dry_run: bool,
    limit: int,
    region: str,
    profile: str | None,
    model_name: str,
    max_tokens: int,
    timeout_seconds: float,
    workdir: Path,
) -> None:
    """Run the medical-wisdom pass-2 pipeline through one-off Bedrock calls."""
    normalized_profile = profile.strip() if profile is not None else None
    if normalized_profile is not None:
        assert normalized_profile, "--profile must be non-empty when set."
    resolved_workdir = workdir.expanduser().resolve()
    resolved_workdir.mkdir(parents=True, exist_ok=True)
    run_log_path = init_logging(level=logging.DEBUG)
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logger.info(
        "Configured main CLI | dry_run=%s | limit=%d | region=%s | model=%s | max_tokens=%d | timeout_seconds=%.1f | workdir=%s | log=%s",
        dry_run,
        limit,
        region,
        model_name,
        max_tokens,
        timeout_seconds,
        resolved_workdir,
        run_log_path,
    )
    ctx.obj = RuntimeConfig(
        dry_run=dry_run,
        limit=limit,
        region=region,
        profile=normalized_profile,
        model_name=model_name,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        workdir=resolved_workdir,
    )


@cli.command("write-smoke-pass1")
@click.option(
    "--output-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/smoke.pass1.jsonl"),
    show_default=True,
    help="Local pass-1 JSONL path to write.",
)
@click.option(
    "--question",
    type=str,
    default=None,
    help="Override the default smoke question with a custom user question.",
)
@click.pass_obj
def write_smoke_pass1(cfg: RuntimeConfig, output_jsonl: Path, question: str | None) -> None:
    records = (
        [
            {
                "dataset": "health_search_qa",
                "id": "smoke_001",
                "question": question.strip(),
            }
        ]
        if question is not None
        else list(SMOKE_RECORDS)
    )
    if cfg.limit > 0:
        records = records[: cfg.limit]
    if question is not None:
        assert records[0]["question"], "--question must be non-empty when set."
    assert records, "No smoke records selected."
    resolved_output_jsonl = output_jsonl.expanduser().resolve()
    logger.info(
        "Preparing smoke pass-1 JSONL | output=%s | logical_records=%d | dry_run=%s",
        resolved_output_jsonl,
        len(records),
        cfg.dry_run,
    )
    if cfg.dry_run:
        click.echo(
            json.dumps(
                {"outputJsonl": str(resolved_output_jsonl), "records": records},
                indent=2,
                sort_keys=True,
            )
        )
        return
    write_jsonl(resolved_output_jsonl, records)
    click.echo(str(resolved_output_jsonl))


@cli.command("annotate-jsonl")
@click.option(
    "--input-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Pass-1 JSONL input to annotate.",
)
@click.option(
    "--output-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Enriched JSONL output path.",
)
@click.option(
    "--failures-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/annotation-failures.jsonl"),
    show_default=True,
    help="Failure log JSONL path.",
)
@click.option(
    "--repair-attempts-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/repair-attempts.jsonl"),
    show_default=True,
    help="Repair-attempt log JSONL path.",
)
@click.option(
    "--run-manifest-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/run-manifest.json"),
    show_default=True,
    help="Run manifest JSON path.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip already-valid rows found in the output JSONL.",
)
@click.pass_obj
def annotate_jsonl(
    cfg: RuntimeConfig,
    input_jsonl: Path,
    output_jsonl: Path,
    failures_jsonl: Path,
    repair_attempts_jsonl: Path,
    run_manifest_json: Path,
    resume: bool,
) -> None:
    records = load_pass1_records(input_jsonl, limit=cfg.limit)
    preview = {
        "inputJsonl": str(input_jsonl.expanduser().resolve()),
        "outputJsonl": str(output_jsonl.expanduser().resolve()),
        "failuresJsonl": str(failures_jsonl.expanduser().resolve()),
        "repairAttemptsJsonl": str(repair_attempts_jsonl.expanduser().resolve()),
        "runManifestJson": str(run_manifest_json.expanduser().resolve()),
        "recordCount": len(records),
        "resume": resume,
        "firstRecord": records[0],
    }
    logger.info(
        "Prepared annotation run | input=%s | output=%s | records=%d | resume=%s | dry_run=%s",
        input_jsonl,
        output_jsonl,
        len(records),
        resume,
        cfg.dry_run,
    )
    if cfg.dry_run:
        click.echo(json.dumps(preview, indent=2, sort_keys=True))
        return
    manifest = asyncio.run(
        annotate_records(
            cfg,
            records=records,
            output_jsonl=output_jsonl,
            failures_jsonl=failures_jsonl,
            repair_attempts_jsonl=repair_attempts_jsonl,
            run_manifest_json=run_manifest_json,
            resume=resume,
        )
    )
    click.echo(json.dumps(manifest, indent=2, sort_keys=True))


@cli.command("validate-enriched-jsonl")
@click.option(
    "--source-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Pass-1 JSONL used as the source of truth.",
)
@click.option(
    "--input-jsonl",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Enriched JSONL to validate.",
)
@click.option(
    "--summary-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/enriched.validation.json"),
    show_default=True,
    help="Validation summary JSON output path.",
)
@click.pass_obj
def validate_enriched_jsonl(
    cfg: RuntimeConfig,
    source_jsonl: Path,
    input_jsonl: Path,
    summary_json: Path,
) -> None:
    if cfg.dry_run:
        click.echo(
            json.dumps(
                {
                    "sourceJsonl": str(source_jsonl.expanduser().resolve()),
                    "inputJsonl": str(input_jsonl.expanduser().resolve()),
                    "summaryJson": str(summary_json.expanduser().resolve()),
                    "limit": cfg.limit,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    source_records = load_pass1_records(source_jsonl, limit=0)
    summary = validate_enriched_output(
        source_records=source_records,
        output_jsonl=input_jsonl,
        limit=cfg.limit,
    )
    resolved_summary_json = summary_json.expanduser().resolve()
    write_json(resolved_summary_json, summary)
    logger.info("Wrote validation summary | path=%s", resolved_summary_json)
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


@cli.command("smoke-annotate")
@click.option(
    "--question",
    type=str,
    default=None,
    help="Optional custom smoke-test question to use instead of the built-in sample.",
)
@click.option(
    "--output-prefix",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("run/smoke"),
    show_default=True,
    help="Prefix for smoke-test artifacts.",
)
@click.pass_obj
def smoke_annotate(cfg: RuntimeConfig, question: str | None, output_prefix: Path) -> None:
    logical_records = (
        [
            {
                "dataset": "health_search_qa",
                "id": "smoke_001",
                "question": question.strip(),
            }
        ]
        if question is not None
        else [SMOKE_RECORDS[0]]
    )
    if cfg.limit > 0:
        logical_records = logical_records[: cfg.limit]
    if question is not None:
        assert logical_records[0]["question"], "--question must be non-empty when set."
    assert logical_records, "No logical smoke records selected."

    resolved_prefix = output_prefix.expanduser().resolve()
    smoke_pass1_path = resolved_prefix.with_suffix(".pass1.jsonl")
    smoke_output_path = resolved_prefix.with_suffix(".enriched.jsonl")
    smoke_failures_path = resolved_prefix.with_suffix(".failures.jsonl")
    smoke_repairs_path = resolved_prefix.with_suffix(".repair-attempts.jsonl")
    smoke_manifest_path = resolved_prefix.with_suffix(".run-manifest.json")
    smoke_validation_path = resolved_prefix.with_suffix(".validation.json")
    logger.info(
        "Running smoke annotation | logical_records=%d | output_prefix=%s | dry_run=%s",
        len(logical_records),
        resolved_prefix,
        cfg.dry_run,
    )
    preview = {
        "smokePass1Path": str(smoke_pass1_path),
        "smokeOutputPath": str(smoke_output_path),
        "smokeFailuresPath": str(smoke_failures_path),
        "smokeRepairsPath": str(smoke_repairs_path),
        "smokeManifestPath": str(smoke_manifest_path),
        "smokeValidationPath": str(smoke_validation_path),
        "recordCount": len(logical_records),
        "firstRecord": logical_records[0],
    }
    if cfg.dry_run:
        click.echo(json.dumps(preview, indent=2, sort_keys=True))
        return

    write_jsonl(smoke_pass1_path, logical_records)
    manifest = asyncio.run(
        annotate_records(
            cfg,
            records=logical_records,
            output_jsonl=smoke_output_path,
            failures_jsonl=smoke_failures_path,
            repair_attempts_jsonl=smoke_repairs_path,
            run_manifest_json=smoke_manifest_path,
            resume=False,
        )
    )
    validation_summary = validate_enriched_output(
        source_records=logical_records,
        output_jsonl=smoke_output_path,
        limit=cfg.limit or len(logical_records),
    )
    write_json(smoke_validation_path, validation_summary)
    click.echo(
        json.dumps(
            {
                "manifest": manifest,
                "validationSummary": validation_summary,
                "smokePass1Path": str(smoke_pass1_path),
                "smokeOutputPath": str(smoke_output_path),
                "smokeFailuresPath": str(smoke_failures_path),
                "smokeRepairsPath": str(smoke_repairs_path),
                "smokeManifestPath": str(smoke_manifest_path),
                "smokeValidationPath": str(smoke_validation_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    cli()
