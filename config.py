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
# 5. SYSTEM PROMPT — STRICT ENTERPRISE RAG BEHAVIOR
# ──────────────────────────────────────────────────────────
# This prompt is intentionally SHORT and directive to prevent
# small models (Phi-3.5-mini, Qwen) from hallucinating extra
# turns, inventing document tags, or simulating conversations.
SYSTEM_PROMPT_TEMPLATE = """You are a highly efficient enterprise AI assistant.

CRITICAL RULES:
1. LANGUAGE: ALWAYS respond in the exact same language the user used (e.g., if Turkish, respond in Turkish).
2. DIRECTNESS: NEVER explain your internal rules. NEVER use parentheses to say "I don't have a source" or "I am chatting normally". Just give the direct, natural answer and stop.

Context:
{context}

Instructions based on the context above:
- If the context is empty, chat normally and politely with the user. Do NOT mention a knowledge base, rules, or lack of sources.
- If context is provided, answer the user's question using ONLY that context. Cite the source filename.
- If context is provided but does not contain the answer, say you don't know — in the user's language.
- Do NOT simulate a conversation. Do NOT generate follow-up questions.
- Give ONE concise answer, then STOP.
"""


# ──────────────────────────────────────────────────────────
# 6. STREAMLIT UI CONFIGURATION
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
