"""API routes. POST /ask and POST /upload for documents."""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import get_settings
from src.models.schemas import AskRequest, AskResponse
from src.agents.router import route_and_answer
from src.ingestion.document_processor import process_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {e.lower() for e in SUPPORTED_EXTENSIONS}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Ask a question. Mode: auto (classify), insight (RAG), or strategic (Decision Agent).
    Returns agent_type, answer, sources, optional reasoning_trace and risk_summary.
    """
    try:
        response = route_and_answer(request)
        # Ensure Pydantic serialization
        return AskResponse(
            agent_type=response.agent_type,
            answer=response.answer,
            sources=response.sources,
            reasoning_trace=response.reasoning_trace,
            risk_summary=response.risk_summary,
        )
    except Exception as e:
        logger.exception("Ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _safe_filename(name: str) -> str:
    """Use basename only and allow only alphanumeric, dash, underscore, dot."""
    base = Path(name).name
    return "".join(c for c in base if c.isalnum() or c in "._- ").strip() or "document"


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(..., description="PDF, TXT, or DOCX files")):
    """
    Upload one or more documents. Files are saved under data/documents and run through ingestion (chunked and added to the vector store).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    settings = get_settings()
    settings.ensure_dirs()
    docs_dir = Path(settings.documents_dir)
    results = []
    total_chunks = 0
    for upload in files:
        name = _safe_filename(upload.filename or "document")
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            results.append({"filename": name, "chunks": 0, "error": f"Unsupported type: {suffix}. Use .pdf, .txt, or .docx"})
            continue
        dest = docs_dir / name
        try:
            content = await upload.read()
            dest.write_bytes(content)
            chunks = process_file(dest)
            total_chunks += chunks
            results.append({"filename": name, "chunks": chunks})
        except Exception as e:
            logger.exception("Upload/ingest failed for %s: %s", name, e)
            results.append({"filename": name, "chunks": 0, "error": str(e)})
    successful = [r for r in results if "error" not in r]
    return {"uploaded": len(successful), "chunks_added": total_chunks, "files": results}
