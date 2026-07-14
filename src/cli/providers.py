import click

from src.cli.infra import infra
from src.cli.models import models


@click.group()
def providers() -> None:
    """Manage inference providers (infra and models)."""


providers.add_command(infra)
providers.add_command(models)
