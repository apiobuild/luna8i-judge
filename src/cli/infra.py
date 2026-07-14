import asyncio

import click


@click.group()
def infra() -> None:
    """Infra provider commands."""


@infra.command()
def refresh() -> None:
    """Fetch live GPU prices and persist to DB."""
    from src.providers.infra_registry import refresh_infra_providers

    providers = asyncio.run(refresh_infra_providers())
    for p in providers:
        spot = f"${p.pricing.spot:.2f}" if p.pricing.spot else "—"
        click.echo(f"{p.name:<40} spot={spot:<10} on_demand=${p.pricing.on_demand:.2f}")
    click.echo(f"\n{len(providers)} providers refreshed.")
