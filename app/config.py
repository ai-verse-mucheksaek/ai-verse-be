from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_path: Path = Path("data/its_me.db")
    storage_dir: Path = Path("data")
    photo_ttl_hours: int = 24
    max_upload_mb: int = 10
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    admin_token: str = ""
    public_base_url: str = "http://localhost:8000"
    share_ttl_minutes: int = 30
    cleanup_interval_minutes: int = 10
    rate_limit_per_minute: int = 60
    max_concurrent_try_ons: int = 2
    log_path: Path = Path("data/app.log")
    vto_provider: str = "mock"
    perfectcorp_api_key: str = ""
    perfectcorp_base_url: str = "https://yce-api-01.makeupar.com"
    perfectcorp_poll_interval_seconds: float = 2
    perfectcorp_poll_timeout_seconds: int = 90

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value):
        return value.split(",") if isinstance(value, str) else value

    @field_validator("vto_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in {"mock", "perfectcorp"}:
            raise ValueError("VTO_PROVIDER must be mock or perfectcorp")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

