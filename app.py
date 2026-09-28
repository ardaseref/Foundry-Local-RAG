# -*- coding: utf-8 -*-
"""
============================================================
app.py -- Streamlit Frontend for the Foundry RAG Assistant
============================================================
Enterprise-grade chat interface built entirely with native
Streamlit components. No custom CSS -- adapts cleanly to
Streamlit's built-in light and dark themes.

Key Architecture:
  - RAGPipeline in st.session_state (models load once)
  - Chat history persists across Streamlit reruns
  - Thread-safe model loading with ScriptRunContext forwarding
  - Streaming responses with typewriter effect

Run with:
    streamlit run app.py

Author:  Enterprise RAG Assistant Project
============================================================
"""

import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from config import CHAT_MODEL, EMBEDDING_MODEL, PAGE_TITLE, TOP_K
from rag_pipeline import RAGPipeline


# ──────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION (must be the FIRST Streamlit call)
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="\U0001f916",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# 2. SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# st.session_state is a persistent dict that survives reruns.
#
# We store:
#   pipeline  -- RAGPipeline instance (models loaded once)
#   messages  -- Chat history [{role, content, sources}, ...]
#   is_ready  -- True once both models are loaded
# ──────────────────────────────────────────────────────────

def init_session_state():
    """Create session-state keys if they don't already exist."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None


# ──────────────────────────────────────────────────────────
# 3. MODEL LOADING (thread-safe)
# ──────────────────────────────────────────────────────────
# The Foundry Local SDK downloads models on a background thread.
# That thread doesn't have Streamlit's ScriptRunContext, which
# causes "missing ScriptRunContext" warnings when the callback
# tries to update a progress bar.
#
# Fix: Before calling initialize(), we capture the current
# thread's ScriptRunContext. Inside the callback (which runs
# on the SDK's internal thread), we re-attach that context
# via add_script_run_ctx(). This makes the progress bar
# update completely thread-safe.
# ──────────────────────────────────────────────────────────

def load_models():
    """
    Load embedding + chat models with a thread-safe progress bar.

    Loading stages mapped to overall progress:
      Embedding download  ->   0 - 25 %
      Embedding load      ->  25 - 50 %
      Chat download       ->  50 - 75 %
      Chat load           ->  75 - 100%
    """
    # Capture Streamlit's execution context from the MAIN thread
    main_ctx = get_script_run_ctx()

    progress_bar = st.sidebar.progress(0, text="Preparing...")

    def on_progress(stage, value=None):
        """Callback invoked by the SDK -- may run on a background thread."""
        # Attach the main thread's context so st calls are safe
        add_script_run_ctx(threading.current_thread(), main_ctx)

        if stage == "embedding_download" and value is not None:
            progress_bar.progress(int(value * 0.25),
                                  text=f"Downloading embedding model... {value:.0f}%")
        elif stage == "embedding_load":
            progress_bar.progress(30, text="Loading embedding model into memory...")
        elif stage == "chat_download" and value is not None:
            progress_bar.progress(50 + int(value * 0.25),
                                  text=f"Downloading chat model... {value:.0f}%")
        elif stage == "chat_load":
            progress_bar.progress(80, text="Loading chat model into memory...")

    try:
        st.session_state.pipeline.initialize(progress_callback=on_progress)
        progress_bar.progress(100, text="All models loaded!")
        st.session_state.is_ready = True
        st.rerun()
    except Exception as e:
        progress_bar.empty()
        st.sidebar.error(f"Failed to load models: {e}")


# ──────────────────────────────────────────────────────────
# 4. SIDEBAR
# ──────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar: model controls, KB stats, config info."""
    with st.sidebar:
        st.title("Controls")

        # ── Model status & load button ────────────────────
        if st.session_state.is_ready:
            st.success("Models loaded", icon="\u2705")
        else:
            st.warning("Models not loaded", icon="\u26a0\ufe0f")
            if st.button("Load Models", type="primary", use_container_width=True):
                load_models()
            return  # Don't render stats until models are ready

        st.divider()

        # ── Knowledge Base statistics ─────────────────────
        st.subheader("Knowledge Base")

        stats = st.session_state.pipeline.get_knowledge_base_stats()

        if stats["total_chunks"] == 0:
            st.warning("Empty. Run `python ingest.py` first.")
            return

        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats["total_chunks"])
        col2.metric("Documents", stats["doc_count"])

        with st.expander("Indexed documents"):
            for doc_name, count in stats["doc_stats"].items():
                st.markdown(f"- **{doc_name}** ({count} chunks)")

        st.divider()

        # ── Configuration info ────────────────────────────
        st.subheader("Configuration")
        st.markdown(f"**Embedding:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**Chat:** `{CHAT_MODEL}`")
        st.markdown(f"**Top-K:** `{TOP_K}` chunks")

        st.divider()
        st.caption("Microsoft Foundry Local SDK")
        st.caption("100% Local \u2022 Zero Cloud Dependency")


# ──────────────────────────────────────────────────────────
# 5. CHAT HISTORY RENDERING
# ──────────────────────────────────────────────────────────

def render_sources(sources):
    """Show retrieved chunks inside a native expander."""
    with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
        for src in sources:
            score_pct = src.score * 100
            preview = src.text[:250] + "..." if len(src.text) > 250 else src.text
            st.markdown(
                f"**{src.source}** &mdash; relevance {score_pct:.1f}%\n\n"
                f"> {preview}"
            )
            st.markdown("")  # spacing


def render_chat_history():
    """Re-draw every message stored in session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])


# ──────────────────────────────────────────────────────────
# 6. STREAMING RESPONSE HANDLER
# ──────────────────────────────────────────────────────────

def handle_user_query(user_input: str):
    """
    Run the full RAG flow: Retrieve -> Augment -> Stream answer.

    1. Save & display the user message
    2. Stream tokens from pipeline.query_stream()
    3. Accumulate text in a placeholder (typewriter effect)
    4. After streaming finishes, render sources below
    5. Save the finished response to chat history

    The key to surviving Streamlit reruns is that ONLY the
    session_state messages list matters. Everything rendered
    here is temporary -- render_chat_history() re-draws it
    all from session_state on each rerun.
    """
    # ── Save & show user message ──────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Stream the assistant response ─────────────────────
    with st.chat_message("assistant"):
        # Use a placeholder for the typewriter streaming effect.
        # This placeholder is replaced with final text after streaming.
        response_placeholder = st.empty()

        full_response = ""
        retrieved_sources = []
        
        # Build chat history from session state (excluding the current user_input we just appended)
        # We take all messages up to the last one
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            # Only include user and assistant roles
            if msg["role"] in ["user", "assistant"]:
                chat_history.append({"role": msg["role"], "content": msg["content"]})

        for token, sources in st.session_state.pipeline.query_stream(user_input, chat_history=chat_history):
            # First token carries the source list
            if sources:
                retrieved_sources = sources

            if token:
                full_response += token
                response_placeholder.markdown(full_response + " \u258c")

        # Final render: replace the placeholder with clean text
        response_placeholder.markdown(full_response)

        # Render sources AFTER the response text
        if retrieved_sources:
            render_sources(retrieved_sources)

    # ── Persist to history ────────────────────────────────
    # This is what render_chat_history() reads on every rerun.
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": retrieved_sources,
    })


# ──────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────

def main():
    """Assemble all components into the final app layout."""
    init_session_state()

    # ── Header ────────────────────────────────────────────
    st.title("Foundry RAG Assistant")
    st.caption("Enterprise Knowledge Base \u2014 Powered by Microsoft Foundry Local (100% Offline)")

    # ── Sidebar ───────────────────────────────────────────
    render_sidebar()

    # ── Chat history ──────────────────────────────────────
    render_chat_history()

    # ── Chat input (disabled until models are ready) ──────
    if st.session_state.is_ready:
        # If we have a pending input, process it now while the input box is locked
        if st.session_state.pending_input:
            user_input = st.session_state.pending_input
            st.session_state.pending_input = None
            st.session_state.is_generating = True
            
            # Render a disabled input box immediately to prevent concurrent submissions
            st.chat_input("Generating response...", disabled=True)
            handle_user_query(user_input)
            
            # Re-enable and refresh
            st.session_state.is_generating = False
            st.rerun()
        else:
            user_input = st.chat_input("Ask a question about your knowledge base...", disabled=st.session_state.is_generating)
            if user_input:
                st.session_state.pending_input = user_input
                st.rerun()
    else:
        st.chat_input("Load the models first (sidebar)...", disabled=True)
        if not st.session_state.messages:
            st.info(
                "Click **Load Models** in the sidebar to initialize the AI engine. "
                "Once loaded, you can ask questions about your indexed knowledge base.",
                icon="\u2139\ufe0f",
            )


if __name__ == "__main__":
    main()
