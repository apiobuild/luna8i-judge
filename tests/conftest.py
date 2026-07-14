from collections.abc import Generator
from unittest.mock import patch

import pytest
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def tmp_db(tmp_path) -> Generator[Session, None, None]:
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def _no_live_infra_fetch():
    """App startup (lifespan) calls get_infra_providers(), which hits RunPod's
    live API when the DB is empty. Block that network call in every test so
    the suite doesn't depend on a third-party API being reachable/well-formed."""
    with patch("src.providers.infra_registry._fetch_runpod", return_value=[]):
        yield
