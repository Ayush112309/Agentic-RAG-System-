# app/ingestion/document_loader.py
"""
Document Loader Module
Handles ingestion of PDF, TXT, and CSV documents.
Supports single files and entire directories.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader,
)

# Configure logger for this module
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading of documents from various formats and sources.

    Supported formats:
        - PDF  (.pdf)  → PyPDFLoader
        - Text (.txt)  → TextLoader
        - CSV  (.csv)  → CSVLoader

    Usage:
        loader = DocumentLoader()
        docs = loader.load_file("path/to/file.pdf")
        all_docs = loader.load_directory("data/samples/")
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".txt": "txt",
        ".csv": "csv",
    }

    def __init__(self) -> None:
        self.loaded_files: List[str] = []
        logger.info("DocumentLoader initialized.")

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single document from a given file path.

        Args:
            file_path: Path to the file to be loaded.

        Returns:
            List of LangChain Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {extension}")
            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        logger.info(f"Loading file: {file_path} (type: {extension})")

        try:
            documents = self._load_by_type(file_path, extension)
            self.loaded_files.append(str(file_path))
            logger.info(f"Successfully loaded {len(documents)} document chunks from {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _load_by_type(self, file_path: Path, extension: str) -> List[Document]:
        """
        Internal dispatcher: routes file to the correct loader based on extension.

        Args:
            file_path: Path object for the file.
            extension: File extension string (e.g., '.pdf').

        Returns:
            List of LangChain Document objects.
        """
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif extension == ".csv":
            loader = CSVLoader(str(file_path))
        else:
            raise ValueError(f"No loader available for extension: {extension}")

        return loader.load()

    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory containing documents.
            recursive: If True, searches subdirectories as well.

        Returns:
            Combined list of LangChain Document objects from all files.
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_documents: List[Document] = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(directory_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping {file_path.name} due to error: {e}")

        logger.info(
            f"Directory load complete. Total documents loaded: {len(all_documents)} "
            f"from {len(self.loaded_files)} files."
        )
        return all_documents

    def load_uploaded_file(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load a document from raw bytes (e.g., Streamlit file upload).
        Saves to a temp path, loads it, then removes the temp file.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename including extension.

        Returns:
            List of LangChain Document objects.
        """
        import tempfile

        extension = Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        # Write bytes to a named temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, prefix="rag_upload_"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            documents = self.load_file(tmp_path)
            # Patch metadata so the source shows the original filename
            for doc in documents:
                doc.metadata["source"] = filename
            return documents
        finally:
            os.unlink(tmp_path)  # Always clean up the temp file

    @property
    def loaded_file_count(self) -> int:
        """Return the number of files loaded so far."""
        return len(self.loaded_files)