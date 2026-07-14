import click


@click.group()
def managed() -> None:
    """Manage API keys for managed inference providers."""


@managed.command("list")
def managed_list() -> None:
    """List configured provider API keys (masked)."""
    from sqlmodel import select

    from src.db import get_session
    from src.providers.managed_model_provider_constants import PROVIDERS_WITH_API_KEY, PROVIDERS_WITH_HOST
    from src.schemas.db import ProviderHost, ProviderKey

    with get_session() as session:
        keys = {r.provider: r.api_key for r in session.exec(select(ProviderKey)).all()}
        hosts = {r.provider: r.host for r in session.exec(select(ProviderHost)).all()}

    for p in PROVIDERS_WITH_API_KEY:
        key = keys.get(p)
        masked = (key[:6] + "*" * min(6, len(key) - 6)) if key else "—"
        click.echo(f"{p:<20} {masked}")
    for p in PROVIDERS_WITH_HOST:
        click.echo(f"{p:<20} {hosts.get(p, '—')}")


@managed.command("set")
@click.argument("provider")
@click.argument("value")
def managed_set(provider: str, value: str) -> None:
    """Set an API key or host URL for a provider (e.g. openai sk-...)."""
    from src.db import get_session
    from src.providers.managed_model_provider_constants import PROVIDERS_WITH_API_KEY, PROVIDERS_WITH_HOST
    from src.schemas.db import ProviderHost, ProviderKey

    with get_session() as session:
        if provider in PROVIDERS_WITH_API_KEY:
            row = session.get(ProviderKey, provider)
            if row:
                row.api_key = value
            else:
                session.add(ProviderKey(provider=provider, api_key=value))
        elif provider in PROVIDERS_WITH_HOST:
            row = session.get(ProviderHost, provider)
            if row:
                row.host = value
            else:
                session.add(ProviderHost(provider=provider, host=value))
        else:
            raise click.ClickException(f"Unknown provider: {provider}")
        session.commit()
    click.echo(f"{provider} updated.")
