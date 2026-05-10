# app/utils/pipeline.py
"""
RAG Pipeline Orchestrator
High-level facade that wires all modules together into a single,
easy-to-use interface.

This is the single entry point for both the Streamlit UI and CLI.

Pipeline Architecture:
  DocumentLoader → TextChunker → EmbeddingManager
       ↓
  VectorStoreManager (ChromaDB)
       ↓
  RAGRetriever ← RAGAgent → MemoryManager
       ↓
  AgentResponse (answer + sources + chunks + scores)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.embeddings.embedding_manager import EmbeddingManager
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.retriever import RAGRetriever
from app.agents.rag_agent import RAGAgent, AgentResponse
from app.memory.memory_manager import MemoryManager
from app.utils.config import Config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Top-level orchestrator for the Agentic RAG system.

    Provides simple methods:
      - ingest_file()        → Load + chunk + embed + store one file
      - ingest_directory()   → Batch ingest all supported files
      - query()              → Ask a question, get AgentResponse
      - reset()              → Clear vector store + memory
      - status()             → System health check

    Usage:
        pipeline = RAGPipeline.from_config()
        pipeline.ingest_file("data/samples/manual.pdf")
        response = pipeline.query("What are the installation steps?")
        print(response.answer)
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize pipeline with configuration.

        Args:
            config: Config instance (use Config.from_env() for production).
        """
        self.config = config

        # Initialize all pipeline components
        logger.info("Initializing RAG Pipeline components...")

        self.document_loader = DocumentLoader()

        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        self.embedding_manager = EmbeddingManager(
            backend=config.embedding_backend,
            model_name=config.embedding_model,
        )

        self.vector_store = VectorStoreManager(
            embedding_manager=self.embedding_manager,
            persist_directory=config.chroma_persist_dir,
            collection_name=config.chroma_collection_name,
        )

        self.retriever = RAGRetriever(
            vector_store_manager=self.vector_store,
            k=config.retrieval_k,
            score_threshold=config.retrieval_score_threshold,
        )

        self.memory = MemoryManager(window_size=config.memory_window_size)

        self.agent = RAGAgent(
            retriever=self.retriever,
            memory_manager=self.memory,
            model_name=config.llm_model,
            temperature=config.llm_temperature,
            use_agent_mode=config.use_agent_mode,
        )

        logger.info("RAG Pipeline initialized successfully.")

    @classmethod
    def from_config(cls) -> "RAGPipeline":
        """
        Create a RAGPipeline from environment configuration.

        Returns:
            Configured RAGPipeline instance.
        """
        config = Config.from_env()
        config.validate()
        return cls(config)

    def ingest_file(self, file_path: Union[str, Path]) -> int:
        """
        Ingest a single file into the vector store.

        Steps: Load → Chunk → Embed → Store

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Number of chunks stored.
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path}")

        # Step 1: Load document
        documents = self.document_loader.load_file(file_path)
        if not documents:
            logger.warning(f"No content loaded from {file_path}")
            return 0

        # Step 2: Chunk
        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning(f"No chunks produced from {file_path}")
            return 0

        # Step 3 & 4: Embed + Store (ChromaDB handles embedding internally)
        added = self.vector_store.add_documents(chunks)

        logger.info(
            f"File ingested: {file_path.name} → "
            f"{len(documents)} pages → {len(chunks)} chunks → "
            f"{added} stored"
        )
        return added

    def ingest_uploaded_file(self, file_bytes: bytes, filename: str) -> int:
        """
        Ingest a file uploaded via Streamlit (raw bytes).

        Args:
            file_bytes: Raw file bytes from Streamlit uploader.
            filename: Original filename with extension.

        Returns:
            Number of chunks stored.
        """
        logger.info(f"Ingesting uploaded file: {filename}")

        documents = self.document_loader.load_uploaded_file(file_bytes, filename)
        if not documents:
            return 0

        chunks = self.chunker.split_documents(documents)
        added = self.vector_store.add_documents(chunks)

        logger.info(
            f"Uploaded file ingested: {filename} → "
            f"{len(documents)} pages → {len(chunks)} chunks"
        )
        return added

    def ingest_directory(
        self,
        directory_path: Union[str, Path] = None,
        recursive: bool = False,
    ) -> dict:
        """
        Ingest all supported files from a directory.

        Args:
            directory_path: Path to the directory. Defaults to config.data_dir.
            recursive: If True, search subdirectories too.

        Returns:
            Dict with ingestion stats.
        """
        directory_path = Path(directory_path or self.config.data_dir)
        logger.info(f"Ingesting directory: {directory_path}")

        documents = self.document_loader.load_directory(directory_path, recursive)
        if not documents:
            return {"files": 0, "chunks": 0, "stored": 0}

        chunks = self.chunker.split_documents(documents)
        stored = self.vector_store.add_documents(chunks)

        stats = {
            "files": self.document_loader.loaded_file_count,
            "chunks": len(chunks),
            "stored": stored,
        }
        logger.info(f"Directory ingestion complete: {stats}")
        return stats

    def query(self, user_query: str) -> AgentResponse:
        """
        Process a user question through the full RAG pipeline.

        Args:
            user_query: Natural language question from the user.

        Returns:
            AgentResponse with answer, sources, and metadata.
        """
        return self.agent.query(user_query)

    def clear_memory(self) -> None:
        """Clear conversation memory (keep documents in vector store)."""
        self.agent.clear_memory()
        logger.info("Conversation memory cleared.")

    def reset(self) -> None:
        """
        Full reset: clear vector store AND memory.
        WARNING: This deletes all ingested documents!
        """
        self.vector_store.reset()
        self.memory.clear()
        logger.info("Full pipeline reset complete.")

    def status(self) -> dict:
        """
        Return system health/status information.

        Returns:
            Dict with component status and stats.
        """
        return {
            "vector_store": {
                "document_count": self.vector_store.document_count(),
                "sources": self.vector_store.get_sources(),
                "persist_dir": self.config.chroma_persist_dir,
            },
            "memory": {
                "turn_count": self.memory.turn_count,
                "message_count": self.memory.message_count,
            },
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_backend": self.config.embedding_backend,
                "chunk_size": self.config.chunk_size,
                "retrieval_k": self.config.retrieval_k,
            },
        }