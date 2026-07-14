import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

import click

from src.services.job_constants import PIPELINE_STEPS

T = TypeVar("T")


def _parse_json_option(value: str | None, label: str, factory: Callable[[Any], T] | None = None) -> T | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise click.ClickException(f"{label} is not valid JSON.")
    if factory is None:
        return parsed
    try:
        return factory(parsed)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"{label} is not valid JSON.") from exc


def _run_pipeline_with_progress(
    job_id: str,
    output_dir: Path | None,
    pipeline_fn: Callable[..., None],
) -> None:
    """Run pipeline_fn with step-status progress printed to stderr."""
    _is_tty = sys.stderr.isatty()
    _last_bar_line: list[str] = [""]
    _current_step: list[str] = [""]
    # model tracking for inference and evaluation: (current_model, model_index, total_models)
    _inference_model: list[tuple[str, int, int]] = [("", 0, 0)]
    _evaluation_model: list[tuple[str, int, int]] = [("", 0, 0)]

    def _render_bar(label: str, completed: int, total: int, model: str) -> None:
        if not total:
            return
        pct = int(completed / total * 100)
        # In non-TTY mode only print at 25% milestones to avoid flooding logs.
        if not _is_tty and pct % 25 != 0:
            return
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        if label == "inference":
            _m, _idx, _total_m = _inference_model[0]
        elif label == "evaluation":
            _m, _idx, _total_m = _evaluation_model[0]
        else:
            _idx, _total_m = 0, 0
        model_tag = f"  [{_idx + 1}/{_total_m}] {model}" if _total_m > 1 else f"  {model}"
        line = f"    [{bar}] {completed}/{total} ({pct}%){model_tag}"
        prev = _last_bar_line[0]
        if line == prev:
            return
        if _is_tty and prev:
            click.echo("\r" + " " * len(prev) + "\r", nl=False, err=True)
        click.echo(("\r" if _is_tty else "") + line, nl=not _is_tty, err=True)
        _last_bar_line[0] = line

    def _flush_bar() -> None:
        if _is_tty and _last_bar_line[0]:
            click.echo("", err=True)
            _last_bar_line[0] = ""

    class _BarFlushHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: ARG002
            _flush_bar()

    _flush_handler = _BarFlushHandler()
    _flush_handler.setLevel(logging.WARNING)
    logging.root.addHandler(_flush_handler)

    def _on_node_start(node_name: str) -> None:
        _flush_bar()
        _current_step[0] = node_name
        label = node_name
        click.echo(f"  {label}…", err=True)

    def _on_node_complete(node_name: str) -> None:
        _flush_bar()
        label = node_name
        if _is_tty:
            click.echo(f"\033[1A\033[2K  \033[32m✓\033[0m {label}", err=True)
        else:
            click.echo(f"  ✓ {label}", err=True)

    def _on_node_failed(node_name: str, error: str) -> None:
        _flush_bar()
        label = node_name
        if _is_tty:
            click.echo(f"\033[1A\033[2K  \033[31m✗\033[0m {label}: {error}", err=True)
        else:
            click.echo(f"  ✗ {label}: {error}", err=True)

    def _on_generate_golden_dataset_progress(completed: int, total: int, model: str) -> None:
        _render_bar("golden", completed, total, model)

    def _on_run_inference_model_load(model: str) -> None:
        _flush_bar()
        click.echo(f"    load {model}…", err=True)

    def _on_run_inference_model_unload(model: str) -> None:
        _flush_bar()
        click.echo(f"    unload {model}…", err=True)

    def _on_run_inference_model_start(model: str, index: int, total_models: int) -> None:
        _flush_bar()
        _inference_model[0] = (model, index, total_models)
        if total_models > 1:
            click.echo(f"    [{index + 1}/{total_models}] run inference {model}…", err=True)
        else:
            click.echo(f"    run inference {model}…", err=True)

    def _on_run_inference_model_complete(
        model: str, index: int, total_models: int, failed: int, total_rows: int
    ) -> None:
        _flush_bar()
        succeeded = total_rows - failed
        suffix = f" ({failed} failed)" if failed else ""
        if _is_tty:
            click.echo(f"    \033[32m✓\033[0m {model}  {succeeded}/{total_rows}{suffix}", err=True)
        else:
            click.echo(f"    ✓ {model}  {succeeded}/{total_rows}{suffix}", err=True)

    def _on_run_inference_progress(completed: int, total: int, model: str) -> None:
        _render_bar("inference", completed, total, model)

    def _on_run_evaluation_model_start(model: str, index: int, total_models: int) -> None:
        _flush_bar()
        _evaluation_model[0] = (model, index, total_models)
        if total_models > 1:
            click.echo(f"    [{index + 1}/{total_models}] {model}…", err=True)
        else:
            click.echo(f"    {model}…", err=True)

    def _on_run_evaluation_model_complete(
        model: str, index: int, total_models: int, failed: int, total_rows: int
    ) -> None:
        _flush_bar()
        succeeded = total_rows - failed
        suffix = f" ({failed} failed)" if failed else ""
        if _is_tty:
            click.echo(f"\033[1A\033[2K    \033[32m✓\033[0m {model}  {succeeded}/{total_rows}{suffix}", err=True)
        else:
            click.echo(f"    ✓ {model}  {succeeded}/{total_rows}{suffix}", err=True)

    def _on_run_evaluation_progress(completed: int, total: int, model: str) -> None:
        _render_bar("evaluation", completed, total, model)

    try:
        pipeline_fn(
            job_id,
            output_dir=output_dir,
            on_node_start=_on_node_start,
            on_node_complete=_on_node_complete,
            on_node_failed=_on_node_failed,
            on_generate_golden_dataset_progress=_on_generate_golden_dataset_progress,
            on_run_inference_progress=_on_run_inference_progress,
            on_run_inference_model_start=_on_run_inference_model_start,
            on_run_inference_model_complete=_on_run_inference_model_complete,
            on_run_inference_model_load=_on_run_inference_model_load,
            on_run_inference_model_unload=_on_run_inference_model_unload,
            on_run_evaluation_model_start=_on_run_evaluation_model_start,
            on_run_evaluation_model_complete=_on_run_evaluation_model_complete,
            on_run_evaluation_progress=_on_run_evaluation_progress,
        )
    finally:
        _flush_bar()
        logging.root.removeHandler(_flush_handler)


@click.group()
def job() -> None:
    """Manage benchmarking jobs."""


@job.command()
@click.option("--file", "file_path", required=True, type=click.Path(exists=True), help="Path to input JSONL file")
def upload_file(file_path: str) -> None:
    """Upload a JSONL file and return an upload_id."""
    from pathlib import Path

    from src.services.jobs import UploadError, save_upload

    raw = Path(file_path).read_bytes()
    try:
        upload_id = save_upload(raw)
    except UploadError as exc:
        raise click.ClickException(str(exc))
    click.echo(json.dumps({"upload_id": upload_id}, indent=2))


@job.command("create")
@click.option("--upload-id", default=None, help="upload_id returned by `job upload-file`")
@click.option(
    "--file",
    "file_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to input JSONL file (alternative to --upload-id)",
)
@click.option("--prompt-template", required=True, help="Prompt template string")
@click.option("--sota-model", default=None, help="SOTA model string (e.g. gemini/gemini-2.0-flash)")
@click.option("--alias", default=None, help="Optional job alias")
@click.option("--output-json-schema", default=None, help="JSON schema string or path to .json file")
@click.option("--eval-fields", default=None, help="JSON array of top-level fields to score")
@click.option("--compare-models", default=None, help="JSON array of {model, params} pairs to compare")
@click.option("--judge-criteria", default=None, help="JSON array of {name, description} judge rubric items")
@click.option(
    "--workload-type",
    default=None,
    type=click.Choice(["classification", "extraction", "summarization"]),
    help="Override workload type detection",
)
@click.option(
    "--modality",
    default=None,
    type=click.Choice(["text", "image", "audio", "video"]),
    help="Override modality detection",
)
@click.option(
    "--projection-by-num-records",
    default="[10000,100000,500000,1000000]",
    show_default=True,
    type=str,
    help="Record counts to project by, as a JSON array of positive integers",
)
@click.option("--target-sla-hours", default=None, type=float, help="Target time-to-completion SLA in hours")
@click.option(
    "--hosting-preference",
    default=None,
    type=click.Choice(["managed", "self_hosted"]),
    help="Pre-select hosting option in scale projection",
)
@click.option(
    "--managed-custom-pricing",
    default=None,
    help='Custom managed pricing JSON: \'{"input_per_1m": 0.5, "output_per_1m": 1.5, "tpm_ceiling": 1000000}\'',
)
@click.option(
    "--self-hosted-custom-pricing",
    default=None,
    help=(
        'Custom self-hosted pricing JSON: \'{"instance_type": "p3.2xlarge", "gpu_count": 1, '
        '"gpu_memory_gb": 16, "tokens_per_sec": 800, "gpu_type": "a100_80gb", '
        '"pricing": {"on_demand": 3.06, "spot": 0.92}}\''
    ),
)
@click.option(
    "--run",
    "run_pipeline_flag",
    is_flag=True,
    default=False,
    help="Run the full pipeline synchronously after submission",
)
@click.option("--limit", default=None, type=int, help="Only process the first N rows (useful for quick tests)")
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Directory for per-model inference output JSONL files (default: data/output/<job_id>/inference/)",
)
@click.option(
    "--html-output",
    "html_output",
    default="report.html",
    type=click.Path(),
    help="Path for the report HTML file (default: <job-output-dir>/report.html). Only used with --run.",
)
def create_job(
    upload_id: str | None,
    file_path: str | None,
    prompt_template: str,
    sota_model: str | None,
    alias: str | None,
    output_json_schema: str | None,
    eval_fields: str | None,
    compare_models: str | None,
    judge_criteria: str | None,
    workload_type: str | None,
    modality: str | None,
    projection_by_num_records: str | None,
    target_sla_hours: float | None,
    hosting_preference: str | None,
    managed_custom_pricing: str | None,
    self_hosted_custom_pricing: str | None,
    run_pipeline_flag: bool,
    limit: int | None,
    output_dir: str | None,
    html_output: str,
) -> None:
    """Create a benchmarking job.

    Provide either --upload-id (from a prior `job upload-file`) or --file to read directly from disk.
    Pass --run to execute the full pipeline synchronously (golden generation + inference + report).
    """
    from pathlib import Path
    from typing import Literal, cast

    from src.agents.job_runner import run_pipeline
    from src.providers.client import get_key_from_env
    from src.schemas.jobs import (
        CustomManagedProviderPricing,
        CustomSelfHostedProviderPricing,
        JudgeCriterion,
        RunEntry,
        SubmissionRequest,
    )
    from src.services.job_submission import SubmissionError, parse_submission, parse_submission_from_upload_id
    from src.services.jobs import create_job

    if upload_id is None and file_path is None:
        raise click.ClickException("Provide either --upload-id or --file.")
    if upload_id is not None and file_path is not None:
        raise click.ClickException("--upload-id and --file are mutually exclusive.")

    # resolve output_json_schema: accept a path to a .json file or inline JSON string
    parsed_schema: dict | None = None
    if output_json_schema is not None:
        schema_path = Path(output_json_schema)
        if schema_path.exists():
            try:
                parsed_schema = json.loads(schema_path.read_text())
            except json.JSONDecodeError:
                raise click.ClickException(f"output_json_schema file '{output_json_schema}' is not valid JSON.")
        else:
            parsed_schema = _parse_json_option(output_json_schema, "output_json_schema")

    parsed_eval_fields = _parse_json_option(eval_fields, "eval_fields")
    parsed_compare_models = (
        _parse_json_option(compare_models, "compare_models", lambda v: [RunEntry(**e) for e in v]) or []
    )
    parsed_judge_criteria = _parse_json_option(
        judge_criteria, "judge_criteria", lambda v: [JudgeCriterion(**c) for c in v]
    )
    parsed_managed_custom_pricing = _parse_json_option(
        managed_custom_pricing, "managed_custom_pricing", lambda v: CustomManagedProviderPricing(**v)
    )
    parsed_self_hosted_custom_pricing = _parse_json_option(
        self_hosted_custom_pricing, "self_hosted_custom_pricing", lambda v: CustomSelfHostedProviderPricing(**v)
    )

    raw = _parse_json_option(projection_by_num_records, "projection_by_num_records")
    if not isinstance(raw, list) or not all(isinstance(v, int) and v > 0 for v in raw):  # type: ignore[union-attr]
        raise click.ClickException(
            "--projection-by-num-records must be a JSON array of positive integers, e.g. '[10000,1000000]'"
        )
    parsed_projection_by_num_records: list[int] = raw

    req = SubmissionRequest(
        prompt_template=prompt_template,
        sota_model=sota_model,
        output_json_schema=parsed_schema,
        eval_fields=parsed_eval_fields,
        compare_models=parsed_compare_models,
        judge_criteria=parsed_judge_criteria,
        workload_type=cast(Literal["classification", "extraction", "summarization"] | None, workload_type),
        modality=cast(Literal["text", "image", "audio", "video"] | None, modality),
        alias=alias,
        projection_by_num_records=parsed_projection_by_num_records,
        target_sla_hours=target_sla_hours,
        hosting_preference=cast(Literal["managed", "self_hosted"] | None, hosting_preference),
        managed_provider_custom_pricing=parsed_managed_custom_pricing,
        self_hosted_provider_custom_pricing=parsed_self_hosted_custom_pricing,
    )

    try:
        if upload_id is not None:
            sub = parse_submission_from_upload_id(upload_id, req)
        else:
            raw = Path(file_path).read_bytes()  # type: ignore[arg-type]
            sub = parse_submission(raw, req)
    except SubmissionError as exc:
        raise click.ClickException(str(exc))

    if limit is not None:
        sub.rows = sub.rows[:limit]

    if html_output is not None and not html_output.endswith(".html"):
        raise click.ClickException("--html-output must end with .html")

    created = create_job(
        rows=sub.rows,
        prompt_template=sub.prompt_template,
        sota_model=sub.sota_model,
        alias=sub.alias,
        workload_type=sub.workload_type,
        input_modality=sub.input_modality,
        detected_workload_details=sub.detected_workload_details,
        output_json_schema=sub.output_json_schema,
        eval_fields=sub.eval_fields,
        compare_models=sub.compare_models,
        judge_criteria=sub.judge_criteria,
        projection_by_num_records=sub.projection_by_num_records,
        target_sla_hours=sub.target_sla_hours,
        hosting_preference=sub.hosting_preference,
        managed_provider_custom_pricing=sub.managed_provider_custom_pricing,
        self_hosted_provider_custom_pricing=sub.self_hosted_provider_custom_pricing,
        golden_generation_config=sub.golden_generation_config,
        output_dir=Path(output_dir) if output_dir else None,
        html_output_filename=Path(html_output).name if html_output else None,
    )
    click.echo(json.dumps({"job_id": created.job_id, "status": created.status, "rows": len(sub.rows)}, indent=2))

    if run_pipeline_flag:
        job_id = created.job_id

        click.echo(f"Running pipeline for job {job_id} …")
        _out_dir = Path(output_dir) if output_dir else None
        try:
            _run_pipeline_with_progress(
                job_id,
                output_dir=_out_dir,
                pipeline_fn=lambda jid, **kw: run_pipeline(
                    jid,
                    get_managed_model_provider_api_key_func=get_key_from_env,
                    **kw,
                ),
            )
        except click.ClickException:
            raise
        click.echo(f"Pipeline complete. job_id={job_id}")


@job.command("get")
@click.argument("job_id")
def get_job(job_id: str) -> None:
    """Show parameters for a job."""
    from src.services.jobs import get_job

    params = get_job(job_id)
    if params is None:
        raise click.ClickException(f"job '{job_id}' not found")

    click.echo(json.dumps(params.model_dump(mode="json"), indent=2))


@job.command("run")
@click.argument("job_id")
@click.option(
    "--step",
    default=None,
    type=click.Choice(PIPELINE_STEPS),
    help="Run only this one step. Omit to resume from the first incomplete step.",
)
@click.option(
    "--compare-models",
    default=None,
    help="JSON array of {model, params} — merges with existing compare_models (only valid with --step "
    "run_compare_models_inference)",
)
@click.option(
    "--run-models",
    default=None,
    help="JSON array of model strings to run — filters which compare_models are executed this invocation "
    "without modifying the job (only valid with --step run_compare_models_inference). "
    "Example: '[\"ollama/llava\"]'",
)
@click.option(
    "--golden-dataset-path",
    default=None,
    type=click.Path(exists=True),
    help="Seed generated_golden_dataset from this JSONL before running",
)
@click.option("--force", is_flag=True, default=False, help="Clear this step's blob + downstream before running")
@click.option(
    "--retry",
    "retry",
    is_flag=True,
    default=False,
    help="Re-run rows that errored in the previous attempt of this step",
)
@click.option("--output", "output_dir", default=None, type=click.Path(), help="Directory for inference JSONL files")
@click.option(
    "--html-output",
    "html_output",
    default="report.html",
    type=click.Path(),
    help="Override the report HTML filename (default: stored on job record)",
)
@click.option(
    "--auto",
    "auto_load_and_unload_ollama_models",
    is_flag=True,
    default=False,
    help="Pull each Ollama model before inference; evict after. Only with --step run_compare_models_inference.",
)
def run_job(
    job_id: str,
    step: str | None,
    compare_models: str | None,
    run_models: str | None,
    golden_dataset_path: str | None,
    force: bool,
    retry: bool,
    output_dir: str | None,
    html_output: str | None,
    auto_load_and_unload_ollama_models: bool,
) -> None:
    """Resume or run a specific step of a benchmarking job.

    Omit --step to resume from the first incomplete step.
    Use --step run_compare_models_inference --force to re-run inference with new models.
    Use --step create_scale_and_cost_projection_report --force to recompute cost projections.
    """
    from src.providers.client import get_key_from_env
    from src.schemas.jobs import RunEntry
    from src.services.jobs import JobNotFoundError, JobRunningError, run_job

    # ------------------------------------------------------------------
    # --compare-models / --run-models: only valid with --step inference
    # ------------------------------------------------------------------
    if compare_models is not None and step != "run_compare_models_inference":
        raise click.ClickException("--compare-models is only valid with --step run_compare_models_inference")
    if run_models is not None and step != "run_compare_models_inference":
        raise click.ClickException("--run-models is only valid with --step run_compare_models_inference")
    parsed_compare_models: list[RunEntry] | None = None
    if compare_models is not None:
        parsed_compare_models = _parse_json_option(
            compare_models, "compare_models", lambda v: [RunEntry(**e) for e in v]
        )

    parsed_run_models: list[str] | None = None
    if run_models is not None:
        parsed_run_models = _parse_json_option(run_models, "run_models")

    # ------------------------------------------------------------------
    # --golden-dataset-path: parse before running
    # ------------------------------------------------------------------
    golden_dataset_rows: list[dict] | None = None
    if golden_dataset_path is not None:
        raw_lines = Path(golden_dataset_path).read_text().splitlines()
        golden_dataset_rows = []
        for i, line in enumerate(raw_lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                golden_dataset_rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise click.ClickException(f"golden-dataset-path line {i} is not valid JSON")
        click.echo(f"Seeded golden dataset from {golden_dataset_path} ({len(golden_dataset_rows)} rows)", err=True)

    if html_output is not None and not html_output.endswith(".html"):
        raise click.ClickException("--html-output must end with .html")

    click.echo(f"Running pipeline for job {job_id} …")
    try:
        _run_pipeline_with_progress(
            job_id,
            output_dir=Path(output_dir) if output_dir else None,
            pipeline_fn=lambda jid, **kw: run_job(
                jid,
                step=step,
                force=force,
                compare_models=[e.model_dump() for e in parsed_compare_models] if parsed_compare_models else None,
                golden_dataset_rows=golden_dataset_rows,
                retry=retry,
                get_managed_model_provider_api_key_func=get_key_from_env,
                run_models=parsed_run_models,
                auto_load_and_unload_ollama_models=auto_load_and_unload_ollama_models,
                html_output_filename=html_output,
                **kw,
            ),
        )
    except JobNotFoundError:
        raise click.ClickException(f"job '{job_id}' not found")
    except JobRunningError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Pipeline complete. job_id={job_id}")


@job.command("list")
@click.option("--status", default=None, help="Filter by status (e.g. queued, completed, failed)")
@click.option("--alias", default=None, help="Filter by alias")
@click.option("--limit", default=100, show_default=True, type=int, help="Max number of jobs to return")
@click.option("--offset", default=0, show_default=True, type=int, help="Pagination offset")
def list_jobs(status: str | None, alias: str | None, limit: int, offset: int) -> None:
    """List jobs, newest first."""
    from src.services.jobs import list_jobs as _list_jobs

    jobs = _list_jobs(status=status, alias=alias, limit=limit, offset=offset)
    click.echo(json.dumps([j.model_dump() for j in jobs], indent=2))


@job.command()
@click.argument("job_id")
@click.option("--watch", is_flag=True, default=False, help="Poll until job completes (Ctrl+C to stop)")
@click.option("--interval", default=2, show_default=True, help="Poll interval in seconds")
def job_status(job_id: str, watch: bool, interval: int) -> None:
    """Check job status."""
    import time

    from src.services.job_constants import TERMINAL_STATUSES as terminal_states
    from src.services.jobs import get_job_status

    while True:
        result = get_job_status(job_id)

        if result is None:
            raise click.ClickException(f"job '{job_id}' not found")

        if watch:
            click.clear()
            click.echo(f"Watching job {job_id} (Ctrl+C to stop)\n")

        click.echo(json.dumps(result.model_dump(), indent=2))

        if not watch or result.status in terminal_states:
            break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
