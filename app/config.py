"""
Application configuration using Pydantic Settings.
Loads environment variables and provides typed configuration.
"""
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Google Cloud Project
    gcp_project_id: str = Field(..., alias="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-central1", alias="GCP_REGION")

    # Google Cloud Storage
    gcs_bucket_name: str = Field(..., alias="GCS_BUCKET_NAME")

    # ── AI Providers ────────────────────────────────────────────────────────
    # OpenAI — primary provider
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # Gemini — fallback provider
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")

    # Shared AI settings
    ai_timeout_seconds: int = Field(default=30, alias="AI_TIMEOUT_SECONDS")
    ai_max_retries: int = Field(default=2, alias="AI_MAX_RETRIES")

    # Firestore
    firestore_collection: str = Field(default="rescue_reports", alias="FIRESTORE_COLLECTION")

    # Google Maps API (optional)
    google_maps_api_key: Optional[str] = Field(default=None, alias="GOOGLE_MAPS_API_KEY")

    # Service Configuration
    max_image_size_mb: int = Field(default=10, alias="MAX_IMAGE_SIZE_MB")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Security CORS
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Rate Limiting
    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def max_image_size_bytes(self) -> int:
        """Convert MB to bytes for validation."""
        return self.max_image_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def use_google_maps(self) -> bool:
        """Check if Google Maps API is configured."""
        return bool(self.google_maps_api_key)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()
