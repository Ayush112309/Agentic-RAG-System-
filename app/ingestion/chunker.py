# app/ingestion/chunker.py
"""
Text Chunking Module
Responsible for splitting large documents into smaller, overlapping chunks
suitable for embedding and retrieval.

Strategy: Recursive Character Text Splitter
- Splits on paragraphs → sentences → words → characters
- Overlap ensures context is preserved across chunk boundaries
"""

import logging
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits documents into overlapping text chunks for embedding.

    Design Decisions:
    - chunk_size=800: Balances context richness vs. embedding precision
    - chunk_overlap=150: ~18% overlap to avoid losing cross-boundary context
    - Recursive splitting respects natural language boundaries (paragraphs first)

    Usage:
        chunker = TextChunker()
        chunks = chunker.split_documents(documents)
    """

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Initialize the chunker with configurable chunk parameters.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Try to split on these separators in order
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of LangChain Documents into smaller chunks.

        Each chunk inherits the metadata of its parent document (source, page, etc.)
        and gets an additional 'chunk_index' metadata field.

        Args:
            documents: List of LangChain Document objects to split.

        Returns:
            List of chunked LangChain Document objects.
        """
        if not documents:
            logger.warning("No documents provided to split.")
            return []

        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.splitter.split_documents(documents)

        # Add chunk index metadata for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            # Clean up whitespace in chunk content
            chunk.page_content = chunk.page_content.strip()

        # Filter out empty chunks that might result from whitespace-only pages
        chunks = [c for c in chunks if len(c.page_content) > 20]

        logger.info(f"Chunking complete: {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Split a raw text string into Document chunks.

        Args:
            text: Raw text content to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of LangChain Document objects.
        """
        if not text.strip():
            logger.warning("Empty text provided to split_text.")
            return []

        metadata = metadata or {}
        chunks = self.splitter.create_documents([text], metadatas=[metadata])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks

    @property
    def config(self) -> dict:
        """Return current chunker configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }