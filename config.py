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
# Phi-4-mini-instruct: 3.8B parameter model with excellent native 
# multilingual support (including Turkish) and reasoning capability.
# Replaces qwen3-4b which suffers from ONNX Runtime GenAI token
# corruption bugs (generating 'ä' and infinite loops) when running locally.
CHAT_MODEL = "phi-4-mini"

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
# 0.65 is tuned to prevent false positives (e.g., greetings or
# jokes matching prompt_engineering.txt) while still catching
# genuinely relevant queries like "What is RAG?".
SIMILARITY_THRESHOLD = 0.65

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
# the threshold. The model acts as a helpful general assistant.
SYSTEM_PROMPT_CHAT = """You are a helpful AI assistant.
Answer the user's question directly and concisely.
Always respond in the same language the user writes in."""

# ── 5b. RAG GROUNDED ANSWERING (documents found) ─────────
# Used when relevant chunks are retrieved from the knowledge base.
# The {context} placeholder is filled with labeled source chunks.
SYSTEM_PROMPT_RAG = """You are a factual answering assistant.
Answer the user's question using ONLY the information in the Context below.
Always respond in the same language the user writes in.
Rules:
- Use ONLY facts stated in the Context. Do NOT add outside knowledge.
- If the Context does not contain the answer, respond exactly: "I cannot answer this based on the provided context."
- Keep your answer concise and well-structured.

Context:
{context}"""

# ──────────────────────────────────────────────────────────
# 6. GENERATION PARAMETERS
# ──────────────────────────────────────────────────────────
# temperature=0.1  → Near-deterministic; avoids the ONNX
#                    empty-stream bug that 0.0 causes with
#                    some frequency_penalty values
# max_tokens=512   → Generous ceiling for detailed RAG answers
# frequency_penalty=0.0 → Disabled; the ONNX runtime silently
#                         fails when this is passed as 0.0,
#                         so we conditionally exclude it
# ──────────────────────────────────────────────────────────
CHAT_TEMPERATURE = 0.2
CHAT_MAX_TOKENS = 512
CHAT_FREQUENCY_PENALTY = 0.0

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
