"""Central configuration - no hardcoded paths. Production-ready settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


def _project_root() -> Path:
    """Resolve project root relative to this file."""
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and .env."""

    # Project paths (derived, not hardcoded)
    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    documents_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "documents")
    chroma_persist_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "chroma_db")

    # HuggingFace
    huggingface_hub_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    llm_model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2", alias="LLM_MODEL")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Chroma
    chroma_collection_name: str = Field(default="leadership_docs", alias="CHROMA_COLLECTION")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # UI
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # Retrieval
    top_k_retrieve: int = Field(default=5, alias="TOP_K_RETRIEVE")
    retrieval_score_threshold: float = Field(default=0.3, alias="RETRIEVAL_SCORE_THRESHOLD")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        """Create data directories if they do not exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
