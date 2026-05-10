# app/embeddings/embedding_manager.py
"""
Embedding Manager Module
Handles creation and management of text embeddings.

Supports two backends:
  1. OpenAI Embeddings (text-embedding-3-small) — requires OPENAI_API_KEY
  2. Sentence Transformers (all-MiniLM-L6-v2) — fully local, no API key needed

The manager automatically falls back to sentence-transformers if OpenAI is unavailable.
"""

import logging
import os
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EmbeddingManager:
    """
    Manages text embedding generation.

    Provides a unified interface regardless of which backend is used.
    ChromaDB and LangChain both accept the returned embedding object directly.

    Usage:
        manager = EmbeddingManager(backend="openai")
        embeddings = manager.get_embeddings()
        vector = manager.embed_query("What is RAG?")
    """

    def __init__(
        self,
        backend: str = EmbeddingBackend.OPENAI,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the embedding manager.

        Args:
            backend: 'openai' or 'sentence_transformers'.
            model_name: Override the default model for the selected backend.
        """
        self.backend = EmbeddingBackend(backend)
        self.model_name = model_name
        self._embeddings = None  # Lazy initialization

        logger.info(f"EmbeddingManager created with backend='{backend}'")

    def get_embeddings(self):
        """
        Return the initialized embedding object (lazy init on first call).

        Returns:
            LangChain-compatible embedding object.
        """
        if self._embeddings is None:
            self._embeddings = self._initialize_embeddings()
        return self._embeddings

    def _initialize_embeddings(self):
        """
        Initialize the embedding backend.

        Returns:
            Configured embedding object.

        Raises:
            ValueError: If the selected backend cannot be initialized.
        """
        if self.backend == EmbeddingBackend.OPENAI:
            return self._init_openai()
        elif self.backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            return self._init_sentence_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_openai(self):
        """Initialize OpenAI embeddings."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set. Falling back to sentence-transformers."
            )
            self.backend = EmbeddingBackend.SENTENCE_TRANSFORMERS
            return self._init_sentence_transformers()

        try:
            from langchain_openai import OpenAIEmbeddings

            model = self.model_name or "text-embedding-3-small"
            embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
            )
            logger.info(f"OpenAI embeddings initialized with model '{model}'")
            return embeddings

        except ImportError:
            logger.warning(
                "langchain-openai not installed. Falling back to sentence-transformers."
            )
            self.backend = EmbeddingBackend.SENTENCE_TRANSFORMERS
            return self._init_sentence_transformers()

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

    def _init_sentence_transformers(self):
        """Initialize local sentence-transformers embeddings (no API key required)."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model = self.model_name or "all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                f"Sentence-Transformers embeddings initialized with model '{model}'"
            )
            return embeddings

        except ImportError as e:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers is required when not using OpenAI. "
                "Install with: pip install sentence-transformers"
            ) from e

        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: The user query to embed.

        Returns:
            List of float values representing the query embedding.
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)

    @property
    def backend_name(self) -> str:
        """Human-readable name of the active backend."""
        return self.backend.value