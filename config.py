"""
Centralized Configuration
All project constants live here. Import this module from any
other file to access model names, paths, and prompt templates.
No magic strings scattered across the codebase.
"""

import os
from pathlib import Path

# Project paths
# Base directory is wherever this config file lives
BASE_DIR = Path(__file__).resolve().parent

# Directory containing source documents for ingestion
DOCS_DIR = BASE_DIR / "data" / "sample_docs"

# SQLite database file path
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "knowledge_base.db"

# Foundry local model configuration
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-4-mini"
APP_NAME = "foundry_rag_enterprise"

# Chunking parameters
# Maximum number of words per chunk
CHUNK_SIZE = 150

# Number of overlapping words between consecutive chunks
# Overlap ensures context continuity at chunk boundaries
CHUNK_OVERLAP = 30

# Retrieval parameters
# Number of top-K chunks to retrieve per query
TOP_K = 3

# Minimum cosine similarity score to consider a chunk relevant.
# 0.38 enables highly technical/fictional cross-lingual queries to pass the threshold.
SIMILARITY_THRESHOLD = 0.38

# System prompts
SYSTEM_PROMPT_CHAT = """You are a helpful AI assistant.
Answer the user's question directly and concisely. If you are unsure about something, say so honestly."""

SYSTEM_PROMPT_RAG = """You are a factual AI assistant.
Use the Context below to answer the user's question accurately. 
If the Context is completely irrelevant, ignore it and chat normally using your own knowledge.

Context:
{context}"""

# Generation parameters
CHAT_TEMPERATURE = 0.1
CHAT_MAX_TOKENS = 512
CHAT_FREQUENCY_PENALTY = 0.0

# Streamlit ui configuration
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
