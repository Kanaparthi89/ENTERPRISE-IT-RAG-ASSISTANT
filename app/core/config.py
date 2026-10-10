
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: repository root, based on this file's location.
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Centralized, validated application configuration."""

    # Application
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM provider
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval and search
    tavily_api_key: SecretStr = SecretStr("")
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"
    top_k: int = Field(default=4, ge=1, le=50)
    max_retries: int = Field(default=2, ge=0, le=10)

    # Application authentication
    admin_api_key: SecretStr = SecretStr("")

    # Storage paths
    audit_db_path: Path = BASE_DIR / "data" / "audit.db"
    upload_dir: Path = BASE_DIR / "uploads"
    sample_kb_dir: Path = BASE_DIR / "data" / "sample_kb"

    # Upload limits
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_upload_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".pdf",
        ".docx",
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    @field_validator("pinecone_index_name", "pinecone_namespace")
    @classmethod
    def validate_non_empty_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This setting cannot be empty.")
        return value

    def ensure_directories(self) -> None:
        """Create required local directories when needed."""
        self.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sample_kb_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
