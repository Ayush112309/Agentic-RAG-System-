# app/utils/config.py
"""
Configuration Module
Centralized, environment-aware configuration for the RAG system.
Loads from .env file and provides typed access to all settings.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env file if it exists (graceful — no error if absent)."""
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(".env file loaded.")
        else:
            logger.info("No .env file found; relying on environment variables.")
    except ImportError:
        logger.warning("python-dotenv not installed; skipping .env loading.")


@dataclass
class Config:
    """
    Application configuration loaded from environment variables.

    Usage:
        config = Config.from_env()
        print(config.openai_api_key)
        print(config.chunk_size)
    """

    # --- LLM Settings ---
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    embedding_backend: str = "openai"         # 'openai' or 'sentence_transformers'
    embedding_model: Optional[str] = None     # None = use backend default

    # --- Vector Store ---
    chroma_persist_dir: str = "vectorstore/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # --- Retrieval ---
    retrieval_k: int = 4                      # Number of chunks to retrieve
    retrieval_score_threshold: float = 0.30   # Min similarity score (0–1)

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Memory ---
    memory_window_size: int = 10

    # --- Paths ---
    data_dir: str = "data"
    log_level: str = "INFO"
    log_file: str = "logs/rag_system.log"

    # --- Agent ---
    use_agent_mode: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create a Config instance populated from environment variables.

        All settings have safe defaults — only OPENAI_API_KEY is required
        for OpenAI mode.

        Returns:
            Config instance.
        """
        _load_dotenv()

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL") or None,
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "vectorstore/chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION", "rag_documents"),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            retrieval_score_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.30")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            memory_window_size=int(os.getenv("MEMORY_WINDOW", "10")),
            data_dir=os.getenv("DATA_DIR", "data"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/rag_system.log"),
            use_agent_mode=os.getenv("USE_AGENT_MODE", "true").lower() == "true",
        )

    def validate(self) -> None:
        """
        Validate the configuration and warn about potential issues.
        Does not raise — logs warnings for non-critical issues.
        """
        if not self.openai_api_key and self.embedding_backend == "openai":
            logger.warning(
                "OPENAI_API_KEY is not set but embedding_backend='openai'. "
                "Will fall back to sentence-transformers."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )

        if not 0.0 <= self.retrieval_score_threshold <= 1.0:
            raise ValueError(
                f"RETRIEVAL_THRESHOLD must be between 0.0 and 1.0, "
                f"got {self.retrieval_score_threshold}"
            )

        logger.info("Configuration validated successfully.")

    def __repr__(self) -> str:
        """Safe string representation (masks API key)."""
        masked_key = (
            f"{self.openai_api_key[:8]}..."
            if self.openai_api_key and len(self.openai_api_key) > 8
            else "(not set)"
        )
        return (
            f"Config("
            f"llm_model={self.llm_model}, "
            f"embedding_backend={self.embedding_backend}, "
            f"chunk_size={self.chunk_size}, "
            f"retrieval_k={self.retrieval_k}, "
            f"openai_api_key={masked_key}"
            f")"
        )