from unittest.mock import patch

from sqlmodel import Session, select

from src.db import get_session, init_db


def test_init_db_creates_jobs_table(tmp_path):
    import sqlite3

    from sqlmodel import create_engine as _ce

    import src.db as db_module

    db_file = tmp_path / "judge.db"
    new_engine = _ce(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    with patch.object(db_module, "_engine", new_engine):
        init_db()

    conn = sqlite3.connect(str(db_file))
    table_names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    assert "jobs" in table_names


def test_init_db_is_idempotent(tmp_path):
    db_file = tmp_path / "judge.db"
    from sqlmodel import create_engine as _ce

    import src.db as db_module

    new_engine = _ce(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    with patch.object(db_module, "_engine", new_engine), patch("src.db.DB_PATH", db_file):
        init_db()
        init_db()  # must not raise


def test_get_session_yields_session(tmp_path):
    db_file = tmp_path / "judge.db"
    from sqlmodel import create_engine as _ce

    import src.db as db_module

    new_engine = _ce(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    with patch.object(db_module, "_engine", new_engine), patch("src.db.DB_PATH", db_file):
        init_db()
        with get_session() as s:
            assert isinstance(s, Session)


def test_tmp_db_fixture_has_jobs_table(tmp_db):
    # tmp_db is a Session from conftest; verify JobConfig table exists by querying it
    from src.schemas.db import JobConfig

    result = tmp_db.exec(select(JobConfig)).all()
    assert result == []
