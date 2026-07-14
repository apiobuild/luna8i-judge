"""
luna8i-judge CLI

Usage:
    python -m src.cli job upload --file data.jsonl
    python -m src.cli job submit --upload-id <id> --prompt-template "Extract: {text}"
    python -m src.cli job status <job_id>
    python -m src.cli providers infra refresh
    python -m src.cli providers models ollama list
"""

import click

from src.cli.jobs import job
from src.cli.providers import providers


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """luna8i-judge CLI"""
    ctx.ensure_object(dict)
    from src.db import init_db

    init_db()


main.add_command(job)
main.add_command(providers)
