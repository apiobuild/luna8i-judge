from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

# Import all SQLModel table models before create_all so every table is registered.
import src.schemas.db as _db_models  # noqa: F401, E402
from src.env import settings

DB_PATH = Path(settings.DATABASE_PATH)
_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(_engine)
    # Additive column migrations — only runs ALTER TABLE when the column is absent.
    with _engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(jobs)"))}
        if "step_status" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN step_status JSON"))
            conn.commit()
        if "output_dir" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN output_dir TEXT"))
            conn.commit()
        if "input_file_jsonl_path" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN input_file_jsonl_path TEXT"))
            conn.commit()
        if "generated_golden_dataset_path" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN generated_golden_dataset_path TEXT"))
            conn.commit()
        if "evaluation_progress" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN evaluation_progress JSON"))
            conn.commit()
        if "evaluating_inference_output_path" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN evaluating_inference_output_path TEXT"))
            conn.commit()
        if "scale_and_cost_projection_report_path" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN scale_and_cost_projection_report_path TEXT"))
            conn.commit()
        if "projection_by_num_records" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN projection_by_num_records JSON"))
            conn.commit()
        else:
            # Migrate existing INTEGER rows to JSON arrays.
            conn.execute(
                text(
                    "UPDATE jobs SET projection_by_num_records = json_array(CAST(projection_by_num_records AS INTEGER))"
                    " WHERE projection_by_num_records IS NOT NULL"
                    "   AND json_valid(projection_by_num_records) = 0"
                )
            )
            conn.commit()
        if "target_sla_hours" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN target_sla_hours REAL"))
            conn.commit()
        if "managed_provider_custom_pricing" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN managed_provider_custom_pricing JSON"))
            conn.commit()
        if "self_hosted_provider_custom_pricing" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN self_hosted_provider_custom_pricing JSON"))
            conn.commit()
        if "html_output_filename" not in existing:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN html_output_filename TEXT"))
            conn.commit()


def load_keys_from_db() -> None:
    from src.providers.managed_model_provider_constants import PROVIDERS_WITH_API_KEY
    from src.schemas.db import ProviderKey

    with get_session() as session:
        rows = session.exec(__import__("sqlmodel").select(ProviderKey)).all()
        for row in rows:
            env_var = PROVIDERS_WITH_API_KEY.get(row.provider)
            if env_var:
                setattr(settings, env_var, row.api_key)


def load_hosts_from_db() -> None:
    from src.providers.managed_model_provider_constants import PROVIDERS_WITH_HOST
    from src.schemas.db import ProviderHost

    with get_session() as session:
        rows = session.exec(__import__("sqlmodel").select(ProviderHost)).all()
        for row in rows:
            env_var = PROVIDERS_WITH_HOST.get(row.provider)
            if env_var:
                setattr(settings, env_var, row.host)


@contextmanager
def get_session():
    with Session(_engine) as session:
        yield session
