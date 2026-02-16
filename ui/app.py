"""Streamlit UI: mode selection, display answer, sources, reasoning trace, risk chart."""

import io
import logging
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
API_BASE = f"http://localhost:{settings.api_port}"


def main():
    st.set_page_config(
        page_title="AI Leadership Agent",
        page_icon="🔷",
        layout="wide",
    )
    st.title("🔷 AI Leadership Insight & Autonomous Decision Agent")
    st.markdown("Ask factual questions (Insight) or strategic questions (Strategic Decision Agent).")

    mode = st.radio(
        "Mode",
        options=["auto", "insight", "strategic"],
        format_func=lambda x: {"auto": "Auto (classify)", "insight": "Insight (RAG)", "strategic": "Strategic (Decision Agent)"}[x],
        horizontal=True,
    )
    question = st.text_area("Question", placeholder="e.g. What is our revenue trend? or Should we expand into Southeast Asia?")
    submit = st.button("Ask")

    if submit and question.strip():
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_BASE}/ask",
                    json={"question": question.strip(), "mode": mode},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is the backend running? Start it with: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`")
                return
            except Exception as e:
                st.error(f"Request failed: {e}")
                return

        agent = data.get("agent_type", "insight")
        st.success(f"Answered by **{agent}** agent")
        st.subheader("Answer")
        st.markdown(data.get("answer", ""))

        sources = data.get("sources") or []
        if sources:
            with st.expander("📎 Sources (data fetched from)", expanded=False):
                # Show file names from metadata; deduplicate while preserving order
                seen = set()
                for s in sources:
                    meta = s.get("metadata", {}) if isinstance(s, dict) else getattr(s, "metadata", {}) or {}
                    file_name = meta.get("source_file") or meta.get("source_path", "—")
                    if isinstance(file_name, str) and "/" in file_name:
                        file_name = file_name.split("/")[-1].split("\\")[-1]
                    if file_name not in seen:
                        seen.add(file_name)
                        score = s.get("score") if isinstance(s, dict) else getattr(s, "score", None)
                        label = file_name + (f" (relevance: {score:.0%})" if score is not None else "")
                        st.markdown(f"• **{label}**")

        trace = data.get("reasoning_trace")
        if trace:
            with st.expander("🔍 Reasoning steps", expanded=False):
                for step in trace:
                    node = step.get("node", step.get("node", ""))
                    summary = step.get("summary", step.get("summary", ""))
                    st.markdown(f"**{node}**: {summary}")

        risk = data.get("risk_summary")
        if risk:
            st.subheader("Risk summary")
            scores = risk.get("scores") or {}
            options = risk.get("options") or []
            if scores:
                import pandas as pd
                df = pd.DataFrame({"Risk score": list(scores.values())}, index=list(scores.keys()))
                st.bar_chart(df)
            if options:
                for opt in options:
                    name = opt.get("name", opt.get("summary", str(opt)))
                    level = opt.get("level", "")
                    score = opt.get("score", "")
                    st.write(f"- **{name}**: level {level}, score {score}")

    elif submit and not question.strip():
        st.warning("Please enter a question.")

    st.sidebar.markdown("### Upload documents")
    st.sidebar.caption("PDF, TXT, or DOCX. Stored under data/documents and ingested into the knowledge base.")
    uploaded = st.sidebar.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        upload_btn = st.sidebar.button("Upload and ingest")
        if upload_btn:
            try:
                r = requests.post(
                    f"{API_BASE}/upload",
                    files=[("files", (f.name, io.BytesIO(f.getvalue()), f.type or "application/octet-stream")) for f in uploaded],
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                st.sidebar.success(f"Uploaded {data.get('uploaded', 0)} file(s), {data.get('chunks_added', 0)} chunks added.")
                for f in data.get("files") or []:
                    name = f.get("filename", "")
                    chunks = f.get("chunks", 0)
                    err = f.get("error")
                    if err:
                        st.sidebar.error(f"**{name}**: {err}")
                    else:
                        st.sidebar.caption(f"• {name}: {chunks} chunks")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Could not connect to the API.")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("- **Insight**: Factual answers from company documents (RAG).")
    st.sidebar.markdown("- **Strategic**: Multi-step reasoning, options, risks, recommendation.")
    st.sidebar.markdown("- **Auto**: Classifier picks the agent.")


if __name__ == "__main__":
    main()
