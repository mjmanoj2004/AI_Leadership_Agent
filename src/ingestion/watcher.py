"""Watchdog-based document monitoring. Auto-ingest new/changed files in documents_dir."""

import logging
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from config import get_settings
from src.ingestion.document_processor import process_file

logger = logging.getLogger(__name__)

_observer: Observer | None = None
_lock = threading.Lock()

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}


class DocumentEventHandler(FileSystemEventHandler):
    """Handle new or modified documents and ingest them."""

    def __init__(self, watch_dir: Path) -> None:
        self.watch_dir = Path(watch_dir)

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        return p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info("New file detected: %s", event.src_path)
            try:
                process_file(Path(event.src_path))
            except Exception as e:
                logger.exception("Watcher ingest failed: %s", e)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info("Modified file detected: %s", event.src_path)
            try:
                process_file(Path(event.src_path))
            except Exception as e:
                logger.exception("Watcher ingest failed: %s", e)


def start_document_watcher() -> None:
    """Start watching documents_dir for new/modified files."""
    global _observer
    with _lock:
        if _observer is not None:
            logger.warning("Document watcher already running")
            return
        settings = get_settings()
        watch_dir = settings.documents_dir
        watch_dir.mkdir(parents=True, exist_ok=True)
        _observer = Observer()
        _observer.schedule(
            DocumentEventHandler(watch_dir),
            str(watch_dir),
            recursive=True,
        )
        _observer.start()
        logger.info("Document watcher started: %s", watch_dir)


def stop_document_watcher() -> None:
    """Stop the document watcher."""
    global _observer
    with _lock:
        if _observer is None:
            return
        _observer.stop()
        _observer.join(timeout=5.0)
        _observer = None
        logger.info("Document watcher stopped")
