import json
import subprocess
from pathlib import Path

import click

from src.providers.managed_model_provider_constants import Provider
from src.services.models import (
    OllamaError,
    OllamaNotConfiguredError,
    OllamaUnreachableError,
    VLLMError,
    VLLMNotConfiguredError,
    VLLMUnreachableError,
)

_COMPOSE_FILE = Path(__file__).parents[3] / "docker-compose.yml"


def _compose(service: str, action: str) -> None:
    """Run `docker compose <action> <service>`."""
    if not _COMPOSE_FILE.exists():
        raise click.ClickException(f"docker-compose.yml not found at {_COMPOSE_FILE}")
    cmd = ["docker", "compose", "-f", str(_COMPOSE_FILE), action, service]
    if action == "up":
        cmd.append("-d")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"`docker compose {action} {service}` exited with code {result.returncode}")


def _run_ollama(fn, *args, **kwargs) -> object:
    try:
        return fn(*args, **kwargs)
    except OllamaNotConfiguredError:
        raise click.ClickException("Ollama is not configured. Set OLLAMA_HOST or start Ollama on localhost:11434.")
    except OllamaUnreachableError:
        raise click.ClickException("Ollama service is unreachable. Is Ollama running?")
    except OllamaError as exc:
        raise click.ClickException(f"Ollama returned {exc.status_code}: {exc.body}")


def _run_vllm(fn, *args, **kwargs) -> object:
    try:
        return fn(*args, **kwargs)
    except VLLMNotConfiguredError:
        raise click.ClickException("VLLM_HOST is not set. Set it in your .env or environment.")
    except VLLMUnreachableError:
        raise click.ClickException("vLLM service is unreachable. Is the vLLM container running?")
    except VLLMError as exc:
        raise click.ClickException(f"vLLM returned {exc.status_code}: {exc.body}")


# ---------- ollama ----------


@click.group()
def ollama() -> None:
    """Manage models in the self-hosted Ollama container."""


@ollama.command("start")
def ollama_start() -> None:
    """Start the Ollama container (docker compose up -d ollama)."""
    _compose(Provider.OLLAMA, "up")
    click.echo(
        "Ollama started.\nRun\n```\nexport OLLAMA_HOST=http://localhost:11434\n```\nto your shell to use Ollama."
    )


@ollama.command("stop")
def ollama_stop() -> None:
    """Stop the Ollama container."""
    _compose(Provider.OLLAMA, "stop")
    click.echo("Ollama stopped.")


@ollama.command("list")
def ollama_list() -> None:
    """List installed Ollama models."""
    from src.services.models import ollama_list_models

    click.echo(json.dumps(_run_ollama(ollama_list_models), indent=2))


@ollama.command("running")
def ollama_running() -> None:
    """List currently loaded Ollama models."""
    from src.services.models import ollama_list_running

    click.echo(json.dumps(_run_ollama(ollama_list_running), indent=2))


@ollama.command("pull")
@click.argument("model")
def ollama_pull(model: str) -> None:
    """Pull (download) a model into Ollama (e.g. qwen2.5)."""
    from src.services.models import ollama_pull_with_output_fn

    _run_ollama(ollama_pull_with_output_fn, model, click.echo)
    click.echo(f"Done: {model} pulled.")


@ollama.command("unload")
@click.argument("model")
def ollama_unload(model: str) -> None:
    """Unload an installed Ollama model."""
    from src.services.models import ollama_unload_model

    click.echo(json.dumps(_run_ollama(ollama_unload_model, model), indent=2))


# ---------- vllm ----------


@click.group()
def vllm() -> None:
    """Manage models in the self-hosted vLLM server."""


@vllm.command("start")
@click.option("--model", default=None, help="HuggingFace model ID to serve (overrides docker-compose default).")
def vllm_start(model: str | None) -> None:
    """Start the vLLM container (docker compose up -d vllm)."""
    if model:
        click.echo("Note: --model override requires editing docker-compose.yml; starting with configured model.")
    _compose(Provider.VLLM, "up")
    click.echo("vLLM started.\nRun\n```\nexport VLLM_HOST=http://localhost:8000\n```\nto your shell to use vLLM.")


@vllm.command("stop")
def vllm_stop() -> None:
    """Stop the vLLM container."""
    _compose(Provider.VLLM, "stop")
    click.echo("vLLM stopped.")


@vllm.command("list")
def vllm_list() -> None:
    """List currently loaded vLLM models."""
    from src.services.models import vllm_list_models

    click.echo(json.dumps(_run_vllm(vllm_list_models), indent=2))


@vllm.command("load")
@click.argument("model")
def vllm_load(model: str) -> None:
    """Load a model into the vLLM server (e.g. Qwen/Qwen2.5-0.5B-Instruct)."""
    from src.services.models import vllm_load_model

    click.echo(json.dumps(_run_vllm(vllm_load_model, model), indent=2))


@vllm.command("unload")
@click.argument("model")
def vllm_unload(model: str) -> None:
    """Unload a model from the vLLM server."""
    from src.services.models import vllm_unload_model

    click.echo(json.dumps(_run_vllm(vllm_unload_model, model), indent=2))


# ---------- hosted ----------


@click.group()
def hosted() -> None:
    """Manage self-hosted inference containers (Ollama, vLLM)."""


hosted.add_command(ollama)
hosted.add_command(vllm)
