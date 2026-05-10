# tests/test_pipeline.py
"""
Test suite for the Agentic RAG System.

Tests cover:
  - DocumentLoader (load_file, load_directory, load_uploaded_file)
  - TextChunker (split_documents, split_text)
  - EmbeddingManager (initialization, fallback)
  - VectorStoreManager (add, search, reset)
  - RAGRetriever (retrieve, threshold filtering)
  - MemoryManager (add, get, clear)
  - RAGPipeline (ingest, query, status)
  - Agent tools (calculator, retrieval)

Run with: pytest tests/test_pipeline.py -v
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_txt_file(temp_dir):
    """Create a sample .txt file for testing."""
    path = temp_dir / "test.txt"
    path.write_text(
        "This is a test document about artificial intelligence. "
        "AI is revolutionizing many industries including healthcare, finance, and education. "
        "Machine learning is a subset of AI that uses data to train models.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample .csv file for testing."""
    path = temp_dir / "test.csv"
    path.write_text(
        "name,price,category\nProduct A,100,Software\nProduct B,200,Hardware\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_documents():
    """Return a list of sample LangChain Documents."""
    return [
        Document(
            page_content="This is the first test document. It contains information about Python.",
            metadata={"source": "test1.txt", "chunk_index": 0},
        ),
        Document(
            page_content="This is the second test document. It talks about machine learning.",
            metadata={"source": "test2.txt", "chunk_index": 1},
        ),
        Document(
            page_content="The third document discusses RAG systems and vector databases.",
            metadata={"source": "test3.txt", "chunk_index": 2},
        ),
    ]


# ─── DocumentLoader Tests ─────────────────────────────────────────────────────

class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def test_load_txt_file(self, sample_txt_file):
        """Should load a .txt file and return non-empty documents."""
        from app.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_file(sample_txt_file)
        assert len(docs) > 0
        assert any("artificial intelligence" in doc.page_content.lower() for doc in docs)

    def test_load_csv_file(self, sample_csv_file):
        """Should load a .csv file and return documents."""
        from app.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_file(sample_csv_file)
        assert len(docs) > 0

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing files."""
        from app.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.txt")

    def test_unsupported_extension_raises(self, temp_dir):
        """Should raise ValueError for unsupported file types."""
        from app.ingestion.document_loader import DocumentLoader
        bad_file = temp_dir / "file.docx"
        bad_file.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(bad_file)

    def test_load_directory(self, temp_dir, sample_txt_file, sample_csv_file):
        """Should load all supported files from a directory."""
        from app.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)
        assert len(docs) > 0
        assert loader.loaded_file_count >= 2

    def test_load_uploaded_file(self):
        """Should handle in-memory file uploads."""
        from app.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        content = b"This is an uploaded text document for testing purposes."
        docs = loader.load_uploaded_file(content, "upload.txt")
        assert len(docs) > 0
        assert docs[0].metadata.get("source") == "upload.txt"


# ─── TextChunker Tests ────────────────────────────────────────────────────────

class TestTextChunker:
    """Tests for the TextChunker class."""

    def test_split_documents(self, sample_documents):
        """Should split documents into chunks."""
        from app.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_chunk_metadata_preserved(self, sample_documents):
        """Should preserve source metadata in chunks."""
        from app.ingestion.chunker import TextChunker
        chunker = TextChunker()
        chunks = chunker.split_documents(sample_documents)
        assert all("source" in chunk.metadata for chunk in chunks)

    def test_chunk_index_added(self, sample_documents):
        """Should add chunk_index to metadata."""
        from app.ingestion.chunker import TextChunker
        chunker = TextChunker()
        chunks = chunker.split_documents(sample_documents)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_empty_documents_returns_empty(self):
        """Should return empty list for empty input."""
        from app.ingestion.chunker import TextChunker
        chunker = TextChunker()
        result = chunker.split_documents([])
        assert result == []

    def test_invalid_overlap_raises(self):
        """Should raise ValueError if overlap >= chunk_size."""
        from app.ingestion.chunker import TextChunker
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_split_text(self):
        """Should split raw text into Document chunks."""
        from app.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "Artificial intelligence is transforming the world. Machine learning enables computers to learn."
        chunks = chunker.split_text(text, metadata={"source": "test"})
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)


# ─── MemoryManager Tests ──────────────────────────────────────────────────────

class TestMemoryManager:
    """Tests for the MemoryManager class."""

    def test_add_and_retrieve_messages(self):
        """Should store and return user/AI messages."""
        from app.memory.memory_manager import MemoryManager
        memory = MemoryManager(window_size=5)
        memory.add_user_message("Hello!")
        memory.add_ai_message("Hi there!")

        history = memory.get_history()
        assert len(history) == 2

    def test_message_count(self):
        """Should track total message count."""
        from app.memory.memory_manager import MemoryManager
        memory = MemoryManager()
        memory.add_user_message("Q1")
        memory.add_ai_message("A1")
        memory.add_user_message("Q2")
        memory.add_ai_message("A2")
        assert memory.message_count == 4
        assert memory.turn_count == 2

    def test_clear(self):
        """Should clear all messages on clear()."""
        from app.memory.memory_manager import MemoryManager
        memory = MemoryManager()
        memory.add_user_message("Test")
        memory.add_ai_message("Response")
        memory.clear()
        assert memory.message_count == 0
        assert len(memory.get_history()) == 0

    def test_get_full_history(self):
        """Should return complete history with roles."""
        from app.memory.memory_manager import MemoryManager
        memory = MemoryManager()
        memory.add_user_message("Who are you?")
        memory.add_ai_message("I am an AI assistant.")

        full = memory.get_full_history()
        assert full[0]["role"] == "user"
        assert full[1]["role"] == "assistant"


# ─── Agent Tools Tests ────────────────────────────────────────────────────────

class TestAgentTools:
    """Tests for individual agent tools."""

    def test_calculator_addition(self):
        """Calculator should handle addition."""
        from app.agents.tools import calculator_tool
        result = calculator_tool.invoke("2 + 3")
        assert "5" in result

    def test_calculator_complex(self):
        """Calculator should handle complex expressions."""
        from app.agents.tools import calculator_tool
        result = calculator_tool.invoke("sqrt(16) * 3")
        assert "12" in result

    def test_calculator_division_by_zero(self):
        """Calculator should handle division by zero gracefully."""
        from app.agents.tools import calculator_tool
        result = calculator_tool.invoke("10 / 0")
        assert "Error" in result

    def test_calculator_invalid_expression(self):
        """Calculator should handle invalid expressions gracefully."""
        from app.agents.tools import calculator_tool
        result = calculator_tool.invoke("import os")
        assert "Error" in result or "invalid" in result.lower()

    def test_document_retrieval_no_retriever(self):
        """Document retrieval should handle uninitialized retriever."""
        from app.agents.tools import document_retrieval_tool, set_retriever
        set_retriever(None)
        result = document_retrieval_tool.invoke("What is AI?")
        assert "not initialized" in result.lower() or "please" in result.lower()


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for the Config class."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from app.utils.config import Config
        config = Config()
        assert config.llm_model == "gpt-4o-mini"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.retrieval_k == 4

    def test_invalid_overlap_raises(self):
        """Should raise ValueError when overlap >= chunk_size."""
        from app.utils.config import Config
        config = Config(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            config.validate()

    def test_invalid_threshold_raises(self):
        """Should raise ValueError for threshold outside 0–1."""
        from app.utils.config import Config
        config = Config(retrieval_score_threshold=1.5)
        with pytest.raises(ValueError, match="RETRIEVAL_THRESHOLD"):
            config.validate()

    def test_from_env_with_mock(self):
        """Should load values from environment variables."""
        from app.utils.config import Config
        with patch.dict(os.environ, {
            "LLM_MODEL": "gpt-4o",
            "CHUNK_SIZE": "512",
            "RETRIEVAL_K": "6",
        }):
            config = Config.from_env()
            assert config.llm_model == "gpt-4o"
            assert config.chunk_size == 512
            assert config.retrieval_k == 6


# ─── Integration: Full Pipeline (mocked LLM) ──────────────────────────────────

class TestRAGPipelineIntegration:
    """
    Integration tests for the full RAG pipeline.
    LLM calls are mocked to avoid API costs during testing.
    """

    @pytest.fixture
    def pipeline_with_mock_llm(self, temp_dir):
        """Create a pipeline with mocked LLM, using local sentence-transformers."""
        # Skip if sentence-transformers not available
        pytest.importorskip("sentence_transformers")

        from app.utils.config import Config
        from app.utils.pipeline import RAGPipeline

        config = Config(
            embedding_backend="sentence_transformers",
            chroma_persist_dir=str(temp_dir / "chroma"),
            chunk_size=200,
            chunk_overlap=30,
            retrieval_k=2,
            retrieval_score_threshold=0.0,  # Accept all results in tests
            use_agent_mode=False,
        )

        pipeline = RAGPipeline(config)
        return pipeline

    def test_ingest_and_query(self, pipeline_with_mock_llm, sample_txt_file):
        """Should ingest a document and retrieve relevant context."""
        pipeline = pipeline_with_mock_llm

        # Ingest
        count = pipeline.ingest_file(sample_txt_file)
        assert count > 0

        # Verify stored
        assert pipeline.vector_store.document_count() > 0

        # Query (chain mode with mock LLM)
        with patch.object(
            pipeline.agent,
            "_chain_query",
            return_value=MagicMock(
                answer="Python is a programming language.",
                sources=["test.txt"],
                retrieved_chunks=[],
                top_relevance_score=0.75,
                used_tools=["document_retrieval_tool"],
                is_grounded=True,
            )
        ):
            response = pipeline.query("What is Python?")
            assert response.answer
            assert response.sources

    def test_status_after_ingest(self, pipeline_with_mock_llm, sample_txt_file):
        """Status should reflect ingested documents."""
        pipeline = pipeline_with_mock_llm
        pipeline.ingest_file(sample_txt_file)
        status = pipeline.status()
        assert status["vector_store"]["document_count"] > 0

    def test_reset_clears_store(self, pipeline_with_mock_llm, sample_txt_file):
        """Reset should clear all stored embeddings."""
        pipeline = pipeline_with_mock_llm
        pipeline.ingest_file(sample_txt_file)
        pipeline.reset()
        assert pipeline.vector_store.document_count() == 0