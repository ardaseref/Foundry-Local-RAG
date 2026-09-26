"""
============================================================
config.py — Centralized Configuration for Foundry RAG Assistant
============================================================
All project constants live here. Import this module from any
other file to access model names, paths, and prompt templates.
No magic strings scattered across the codebase.
============================================================
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────────────────
# Base directory is wherever this config file lives
BASE_DIR = Path(__file__).resolve().parent

# Directory containing source documents for ingestion
DOCS_DIR = BASE_DIR / "data" / "sample_docs"

# SQLite database file path
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "knowledge_base.db"

# ──────────────────────────────────────────────────────────
# 2. FOUNDRY LOCAL MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────
# Embedding model — converts text into numerical vectors
# qwen3-embedding-0.6b is small, fast, and effective for RAG
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

# Chat model — generates natural language answers
# phi-3.5-mini offers excellent quality for its size (~3.8B params)
CHAT_MODEL = "phi-3.5-mini"

# Application name registered with Foundry Local SDK
APP_NAME = "foundry_rag_enterprise"

# ──────────────────────────────────────────────────────────
# 3. CHUNKING PARAMETERS
# ──────────────────────────────────────────────────────────
# Maximum number of words per chunk
CHUNK_SIZE = 300

# Number of overlapping words between consecutive chunks
# Overlap ensures context continuity at chunk boundaries
CHUNK_OVERLAP = 50

# ──────────────────────────────────────────────────────────
# 4. RETRIEVAL PARAMETERS
# ──────────────────────────────────────────────────────────
# Number of top-K chunks to retrieve per query
TOP_K = 3

# Minimum cosine similarity score to consider a chunk relevant.
# 0.55 filters out noise (e.g., greetings matching random chunks)
# while still catching genuinely related content.
SIMILARITY_THRESHOLD = 0.55

# ──────────────────────────────────────────────────────────
# 5. SYSTEM PROMPTS — TWO-PROMPT ARCHITECTURE
# ──────────────────────────────────────────────────────────
# Instead of one prompt with If/Else logic (which overwhelms
# small models), we use TWO separate prompts selected at
# runtime based on whether retrieved context exists.
#
# CRITICAL DESIGN RULE:
#   SYSTEM_PROMPT_CHAT must contain ZERO mentions of
#   "context", "sources", "knowledge base", or "documents".
#   This prevents the model from hallucinating fake sources
#   during casual conversation.
# ──────────────────────────────────────────────────────────

# ── 5a. CASUAL CHAT (no relevant documents found) ────────
# Used when the similarity search returns no chunks above
# the threshold. The model acts as a simple friendly assistant.
SYSTEM_PROMPT_CHAT = """You are a friendly and helpful AI assistant.
Always respond in the same language the user writes in.
Give one concise, direct answer and then stop.
Do not generate follow-up questions.
Do not simulate a conversation."""

# ── 5b. RAG GROUNDED ANSWERING (documents found) ─────────
# Used when relevant chunks are retrieved from the knowledge base.
# The {context} placeholder is filled with labeled source chunks.
SYSTEM_PROMPT_RAG = """You are a knowledgeable assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I cannot answer this based on the provided context."

Context:
{context}"""

# ──────────────────────────────────────────────────────────
# 6. GENERATION PARAMETERS — DETERMINISTIC OUTPUT
# ──────────────────────────────────────────────────────────
# These are applied to the Foundry SDK's ChatClientSettings
# to enforce deterministic, bounded generation.
#
# temperature=0.0  → Greedy decoding; absolute determinism
# max_tokens=256   → Hard ceiling prevents runaway loops
#                    (a concise RAG answer fits in ~150 tokens;
#                     256 allows headroom without letting loops
#                     burn 500 tokens of garbage)
# frequency_penalty=0.3 → Penalizes repeated tokens directly
# ──────────────────────────────────────────────────────────
CHAT_TEMPERATURE = 0.0
CHAT_MAX_TOKENS = 256
CHAT_FREQUENCY_PENALTY = 0.3

# ──────────────────────────────────────────────────────────
# 7. STREAMLIT UI CONFIGURATION
# ──────────────────────────────────────────────────────────
# Page title shown in the browser tab
PAGE_TITLE = "Foundry RAG Assistant"

# Subtitle shown in the sidebar
PAGE_SUBTITLE = "Enterprise Knowledge Base"

# Maximum number of messages to display in chat history
MAX_CHAT_HISTORY = 50


def ensure_directories():
    """Create required directories if they don't exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
