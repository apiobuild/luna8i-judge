from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DB_DIR: str = "./db"
    DATABASE_FILENAME: str = "local.db"

    @property
    def DATABASE_PATH(self) -> str:
        return str(Path(self.DB_DIR) / self.DATABASE_FILENAME)

    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    TOGETHER_API_KEY: str | None = None
    FIREWORKS_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    DASHSCOPE_API_KEY: str | None = None

    DEFAULT_SOTA_MODEL: str = "openai/gpt-4o"
    VLLM_HOST: str | None = None
    OLLAMA_HOST: str | None = None

    DATA_DIR: str = "./"
    LOCAL_MODE: bool = False


settings = Settings()
