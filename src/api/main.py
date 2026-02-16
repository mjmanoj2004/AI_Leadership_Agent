"""FastAPI application entry. Config-driven, logging, error handling."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from src.api.routes import router
from src.ingestion.watcher import start_document_watcher, stop_document_watcher

# Logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start document watcher on startup; stop on shutdown."""
    try:
        start_document_watcher()
        logger.info("Document watcher started")
    except Exception as e:
        logger.warning("Could not start document watcher: %s", e)
    yield
    stop_document_watcher()
    logger.info("Document watcher stopped")


app = FastAPI(
    title="AI Leadership Insight & Autonomous Decision Agent",
    description="Dual-agent system: Insight (RAG) and Strategic Decision (LangGraph)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="", tags=["ask"])


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
