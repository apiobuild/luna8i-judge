import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.db import init_db, load_hosts_from_db
from src.providers.infra_registry import get_infra_providers
from src.routers.jobs import jobs_router
from src.routers.providers import providers_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_hosts_from_db()
    await get_infra_providers()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(jobs_router)
app.include_router(providers_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


_static = Path(__file__).parent.parent / "static"
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
