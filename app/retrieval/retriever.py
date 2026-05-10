# app/retrieval/retriever.py
"""
RAG Retriever Module
Handles semantic retrieval and prepares clean grounded context
for the LLM.

Improved for local LLMs (Ollama / llama3):
- Cleaner context formatting
- Better grounding
- Less metadata confusion
- Stronger hallucination prevention
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Structured result from a retrieval operation.
    """

    query: str
    documents: List[Tuple[Document, float]]
    context: str
    sources: List[str]
    top_score: float

    @property
    def has_results(self) -> bool:
        return len(self.documents) > 0

    @property
    def formatted_sources(self) -> str:
        return "\n".join(f"• {s}" for s in self.sources)


class RAGRetriever:
    """
    Main retrieval pipeline.

    Responsibilities:
    - semantic retrieval
    - threshold filtering
    - context assembly
    - source attribution
    """

    DEFAULT_SCORE_THRESHOLD = 0.30
    DEFAULT_K = 4

    def __init__(
        self,
        vector_store_manager,
        k: int = DEFAULT_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:

        self.vs_manager = vector_store_manager
        self.k = k
        self.score_threshold = score_threshold

        logger.info(
            f"RAGRetriever initialized: "
            f"k={k}, score_threshold={score_threshold}"
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full retrieval pipeline.
        """

        logger.info(f"Retrieving context for query: '{query[:80]}'")

        # Retrieve from vector store
        raw_results = self.vs_manager.similarity_search(
            query=query,
            k=self.k,
            score_threshold=self.score_threshold,
        )

        # Extra safety filtering
        filtered = [
            (doc, score)
            for doc, score in raw_results
            if score >= self.score_threshold
        ]

        if not filtered:
            logger.info(
                "No results above threshold. "
                "Returning empty RetrievalResult."
            )

            return RetrievalResult(
                query=query,
                documents=[],
                context="",
                sources=[],
                top_score=0.0,
            )

        # Build clean LLM context
        context = self._assemble_context(filtered)

        # Extract sources
        sources = self._extract_sources(filtered)

        top_score = max(score for _, score in filtered)

        logger.info(
            f"Retrieved {len(filtered)} chunks. "
            f"Top score: {top_score:.3f}. "
            f"Sources: {sources}"
        )

        return RetrievalResult(
            query=query,
            documents=filtered,
            context=context,
            sources=sources,
            top_score=top_score,
        )

    def _assemble_context(
        self,
        results: List[Tuple[Document, float]]
    ) -> str:
        """
        Build clean context for the LLM.

        IMPORTANT:
        Local models like llama3 perform much better with:
        - simple formatting
        - minimal metadata
        - clearly separated sources/content

        Avoid:
        [Chunk 1 | Relevance: ...]
        because local models get confused by this.
        """

        formatted_chunks = []

        for i, (doc, score) in enumerate(results, start=1):

            source = os.path.basename(
                doc.metadata.get("source", "Unknown")
            )

            page = doc.metadata.get("page")

            page_info = ""
            if isinstance(page, int):
                page_info = f" (Page {page + 1})"

            content = doc.page_content.strip()

            chunk_text = f"""
SOURCE: {source}{page_info}

CONTENT:
{content}
"""

            formatted_chunks.append(chunk_text.strip())

        final_context = "\n\n----------------------\n\n".join(
            formatted_chunks
        )

        return final_context

    def _extract_sources(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[str]:
        """
        Extract unique source filenames.
        """

        sources = set()

        for doc, _ in results:
            source = doc.metadata.get("source", "Unknown")
            sources.add(os.path.basename(source))

        return sorted(sources)

    def update_config(
        self,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Dynamically update retrieval settings.
        """

        if k is not None:
            self.k = k
            logger.info(f"Retriever k updated to {k}")

        if score_threshold is not None:
            self.score_threshold = score_threshold
            logger.info(
                f"Score threshold updated to {score_threshold}"
            )