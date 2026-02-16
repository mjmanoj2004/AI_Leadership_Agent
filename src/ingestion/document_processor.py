"""Process documents (PDF, DOCX, TXT) and add to Chroma. No hardcoded paths."""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store, invalidate_corpus_cache

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Chunking: prefer paragraph/sentence boundaries so retrieved excerpts have complete meaning.
# Separators tried in order: paragraph, line, sentence end, space, char.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
)


def _load_document(file_path: Path) -> List[Document]:
    """Load a single file into LangChain Documents."""
    suffix = file_path.suffix.lower()
    path_str = str(file_path)

    if suffix == ".pdf":
        loader = PyPDFLoader(path_str)
    elif suffix == ".txt":
        loader = TextLoader(path_str, encoding="utf-8", autodetect_encoding=True)
    elif suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(path_str)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    for d in docs:
        d.metadata["source_file"] = file_path.name
        d.metadata["source_path"] = path_str
    return docs


def process_file(file_path: Path) -> int:
    """Load, split, and add one file to the vector store. Returns number of chunks added."""
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning("Skipping unsupported file: %s", file_path)
        return 0

    try:
        raw_docs = _load_document(file_path)
        chunks = TEXT_SPLITTER.split_documents(raw_docs)
        if not chunks:
            return 0
        store = get_vector_store()
        store.add_documents(chunks)
        invalidate_corpus_cache()
        logger.info("Ingested %s: %d chunks", file_path.name, len(chunks))
        return len(chunks)
    except Exception as e:
        logger.exception("Failed to process %s: %s", file_path, e)
        raise


def process_directory(directory: Path) -> int:
    """Process all supported files in directory. Returns total chunks added."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            total += process_file(path)
    return total
