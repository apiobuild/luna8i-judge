import click

from src.cli.hosted import hosted
from src.cli.managed import managed


@click.group()
def models() -> None:
    """Manage models across managed and self-hosted providers."""


models.add_command(managed)
models.add_command(hosted)
