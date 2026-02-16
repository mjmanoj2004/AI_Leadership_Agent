"""Ingest documents from data/documents into Chroma. Run from project root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings
from src.ingestion.document_processor import process_directory

if __name__ == "__main__":
    settings = get_settings()
    settings.ensure_dirs()
    docs_dir = settings.documents_dir
    print(f"Ingesting from {docs_dir} ...")
    n = process_directory(docs_dir)
    print(f"Done. Ingested {n} chunks.")
