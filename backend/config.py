# backend/config.py
# ============================================================
# All app settings come from environment variables.
# Pydantic reads them automatically, so locally you set them
# in .env, in Docker via docker-compose environment:, and in
# AWS via EC2 user-data or SSM Parameter Store.
# ============================================================

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    environment: str = "development"
    secret_key: str = "dev-secret-key-change-in-production"
    debug: bool = True

    # Database
    database_url: str = "postgresql://classpulse_user:password@localhost:5432/classpulse"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # ML
    model_path: str = "ml_pipeline/models/saved_models"

    # CORS
    allowed_origins: str = "http://localhost,http://localhost:80,http://localhost:8080"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance (reads .env once)."""
    return Settings()
