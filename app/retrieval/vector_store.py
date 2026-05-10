# app/retrieval/vector_store.py
"""
Vector Store Module
Manages ChromaDB for persistent embedding storage and retrieval.

ChromaDB was chosen because:
  - Simple setup (no external server needed for local mode)
  - Persistent storage out of the box
  - Native LangChain integration
  - Supports metadata filtering
  - Open-source and actively maintained
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# Default persistence directory for ChromaDB
DEFAULT_PERSIST_DIR = "vectorstore/chroma_db"
DEFAULT_COLLECTION_NAME = "rag_documents"


class VectorStoreManager:
    """
    Manages the ChromaDB vector store.

    Responsibilities:
    - Adding document chunks to the vector store
    - Persisting the store to disk
    - Loading an existing store
    - Performing similarity search with scores
    - Resetting/clearing the store

    Usage:
        manager = VectorStoreManager(embedding_manager)
        manager.add_documents(chunks)
        results = manager.similarity_search("What is AI?", k=4)
    """

    def __init__(
        self,
        embedding_manager,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """
        Initialize the vector store manager.

        Args:
            embedding_manager: An EmbeddingManager instance.
            persist_directory: Path where ChromaDB stores its data.
            collection_name: Name of the ChromaDB collection.
        """
        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None

        # Ensure the persistence directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"VectorStoreManager initialized. "
            f"Collection: '{collection_name}', "
            f"Persist dir: '{persist_directory}'"
        )

    @property
    def vectorstore(self) -> Chroma:
        """
        Lazily load or create the ChromaDB vector store.

        Returns:
            Chroma vector store instance.
        """
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create()
        return self._vectorstore

    def _load_or_create(self) -> Chroma:
        """
        Load existing ChromaDB store or create a new one.

        Returns:
            Chroma instance connected to the persistence directory.
        """
        embeddings = self.embedding_manager.get_embeddings()
        try:
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
            count = store._collection.count()
            logger.info(
                f"ChromaDB loaded. Collection '{self.collection_name}' "
                f"has {count} existing embeddings."
            )
            return store
        except Exception as e:
            logger.error(f"Failed to load/create ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add document chunks to the vector store.

        Deduplicates by checking if the store already has embeddings
        for the same source file before adding.

        Args:
            documents: List of chunked LangChain Documents.

        Returns:
            Number of chunks successfully added.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return 0

        logger.info(f"Adding {len(documents)} chunks to ChromaDB...")
        try:
            self.vectorstore.add_documents(documents)
            count = self.vectorstore._collection.count()
            logger.info(
                f"Documents added. Total embeddings in store: {count}"
            )
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with relevance scores.

        Scores are cosine similarity values in range [0, 1].
        Higher is more relevant.

        Args:
            query: The user's question or search query.
            k: Number of top results to return.
            score_threshold: Minimum similarity score to include a result.

        Returns:
            List of (Document, score) tuples sorted by descending relevance.
        """
        if not query.strip():
            logger.warning("Empty query provided.")
            return []

        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
            )

            # Filter by minimum score threshold
            filtered = [
                (doc, score)
                for doc, score in results
                if score >= score_threshold
            ]

            logger.info(
                f"Similarity search for '{query[:50]}...' → "
                f"{len(filtered)} results (threshold={score_threshold})"
            )
            return filtered

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """
        Return a LangChain-compatible retriever object.

        Args:
            k: Number of documents to retrieve per query.
            search_type: 'similarity' or 'mmr' (maximal marginal relevance).

        Returns:
            LangChain retriever.
        """
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    def document_count(self) -> int:
        """Return the current number of stored embeddings."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """
        Delete all documents from the collection.
        Useful for testing or re-ingestion workflows.
        """
        try:
            self.vectorstore.delete_collection()
            self._vectorstore = None  # Force re-init on next access
            logger.info(f"Collection '{self.collection_name}' has been reset.")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            raise

    def get_sources(self) -> List[str]:
        """
        Return a list of unique source files currently in the vector store.

        Returns:
            List of unique source file paths/names.
        """
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            sources = set()
            for meta in results.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception as e:
            logger.warning(f"Could not retrieve sources: {e}")
            return []