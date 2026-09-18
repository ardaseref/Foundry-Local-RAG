"""
============================================================
ingest.py — Data Ingestion & Embedding Pipeline
============================================================
This script is the FIRST step in our RAG pipeline. It does 3 things:

  1. READS documents (text & PDF files) from the data/sample_docs/ folder
  2. CHUNKS them into smaller passages (~300 words each)
  3. EMBEDS each chunk using Microsoft Foundry Local's embedding model
  4. SAVES everything into a local SQLite database

Think of this as "loading the brain" of our AI assistant.
After running this script, the knowledge base is ready for queries.

Usage:
    python ingest.py

Author:  Enterprise RAG Assistant Project
============================================================
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

# PyPDF2 is optional — only needed if you have PDF documents
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[INFO] PyPDF2 not installed. PDF ingestion disabled. Only .txt files will be processed.")

# Foundry Local SDK — Microsoft's on-device AI runtime
from foundry_local_sdk import Configuration, FoundryLocalManager

# Import our centralized configuration (model names, paths, etc.)
from config import (
    APP_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DOCS_DIR,
    EMBEDDING_MODEL,
    ensure_directories,
)


# ──────────────────────────────────────────────────────────
# STEP 1: DOCUMENT READING
# ──────────────────────────────────────────────────────────
# These functions read raw text from .txt and .pdf files.
# Each function returns a single string with all the text
# content from the file.
# ──────────────────────────────────────────────────────────

def read_text_file(file_path: Path) -> str:
    """
    Read a plain text (.txt) file and return its contents.

    Args:
        file_path: Path to the .txt file

    Returns:
        The full text content of the file as a string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(file_path: Path) -> str:
    """
    Read a PDF file and extract all text from every page.

    We use PyPDF2 to iterate through each page and combine
    the extracted text into a single string. This works well
    for text-based PDFs but may not handle scanned documents
    (which would need OCR).

    Args:
        file_path: Path to the .pdf file

    Returns:
        The combined text from all pages as a string
    """
    if not PDF_SUPPORT:
        print(f"  [SKIP] Cannot read PDF '{file_path.name}' — PyPDF2 not installed.")
        return ""

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Scan the documents directory and read all supported files.

    This function looks for .txt and .pdf files in the specified
    directory, reads their content, and returns a list of document
    dictionaries with the filename and text.

    Args:
        docs_dir: Path to the directory containing documents

    Returns:
        A list of dicts, each with 'source' (filename) and 'text' keys.
        Example: [{"source": "foundry_local.txt", "text": "Microsoft Foundry..."}]
    """
    documents = []

    # Get all .txt and .pdf files, sorted alphabetically for consistency
    supported_extensions = [".txt", ".pdf"]
    files = sorted([
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])

    if not files:
        print(f"\n[ERROR] No .txt or .pdf files found in: {docs_dir}")
        print("  Please add documents to this folder and run again.")
        sys.exit(1)

    print(f"\n[DIR] Found {len(files)} document(s) in '{docs_dir.name}/':")

    for file_path in files:
        # Choose the right reader based on file extension
        if file_path.suffix.lower() == ".txt":
            text = read_text_file(file_path)
        elif file_path.suffix.lower() == ".pdf":
            text = read_pdf_file(file_path)
        else:
            continue  # Skip unsupported files

        # Only add non-empty documents
        if text.strip():
            word_count = len(text.split())
            documents.append({
                "source": file_path.name,
                "text": text.strip()
            })
            print(f"  [OK] {file_path.name} ({word_count} words)")
        else:
            print(f"  [WARN] {file_path.name} (empty -- skipped)")

    return documents


# ──────────────────────────────────────────────────────────
# STEP 2: TEXT CHUNKING
# ──────────────────────────────────────────────────────────
# Large documents need to be split into smaller pieces
# called "chunks" before we can embed them. Why?
#
#   - Embedding models work best with shorter text passages
#   - Smaller chunks = more precise retrieval results
#   - Each chunk becomes a searchable unit in our database
#
# We use a "sliding window" approach with overlap to ensure
# no important context is lost at chunk boundaries.
# ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` words.

    How it works (visual example with chunk_size=5, overlap=2):

        Original text:  [A B C D E F G H I J K L]

        Chunk 1:        [A B C D E]
        Chunk 2:            [D E F G H]       ← overlaps by 2 words
        Chunk 3:                [G H I J K]   ← overlaps by 2 words
        Chunk 4:                    [J K L]   ← final chunk (may be shorter)

    The overlap ensures that sentences spanning chunk boundaries
    are captured in at least one chunk, preventing lost context.

    Args:
        text:       The full text to split into chunks
        chunk_size: Maximum number of words per chunk (default: 300)
        overlap:    Number of words to overlap between chunks (default: 50)

    Returns:
        A list of text chunk strings
    """
    # Split the text into individual words
    words = text.split()

    # If the text is shorter than one chunk, return it as-is
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(words):
        # Take a window of 'chunk_size' words starting from 'start'
        end = start + chunk_size
        chunk_words = words[start:end]

        # Rejoin words into a text string for this chunk
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:  # Only add non-empty chunks
            chunks.append(chunk_text)

        # Move the window forward by (chunk_size - overlap) words
        # This creates the overlapping effect
        start += chunk_size - overlap

        # Safety check: avoid infinite loop if overlap >= chunk_size
        if chunk_size - overlap <= 0:
            break

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Take a list of full documents and split each into chunks.

    Each chunk preserves metadata about which document it came from
    and its position (index) within that document. This metadata
    is crucial for source citations in the RAG pipeline.

    Args:
        documents: List of dicts with 'source' and 'text' keys

    Returns:
        A list of chunk dicts, each with:
          - 'source':      Original filename
          - 'chunk_index': Position of this chunk within the document (0-based)
          - 'text':        The chunk's text content
    """
    all_chunks = []

    print(f"\n[CHUNK] Chunking documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}):")

    for doc in documents:
        # Split this document's text into chunks
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk
            })

        print(f"  [DOC] {doc['source']} -> {len(chunks)} chunk(s)")

    print(f"\n  Total chunks created: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────────────────
# STEP 3: EMBEDDING GENERATION (using Foundry Local SDK)
# ──────────────────────────────────────────────────────────
# Embeddings are numerical vectors (lists of numbers) that
# represent the MEANING of text. Similar texts will have
# similar vectors — this is what enables semantic search.
#
# We use Microsoft Foundry Local to run the embedding model
# entirely on your device. No internet connection needed!
# ──────────────────────────────────────────────────────────

def initialize_embedding_model():
    """
    Initialize the Foundry Local SDK and load the embedding model.

    This function:
      1. Creates a Configuration with our app name
      2. Initializes the FoundryLocalManager (the SDK's core)
      3. Downloads the embedding model (first run only — cached after)
      4. Loads the model into memory for inference
      5. Returns the embedding client ready to generate vectors

    Returns:
        A tuple of (embedding_client, embedding_model) so we can
        generate embeddings and later unload the model to free memory.
    """
    print("\n[AI] Initializing Foundry Local SDK...")

    # Step 1: Configure the SDK with our application name
    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Step 2: Get the embedding model from the catalog
    print(f"  Model: {EMBEDDING_MODEL}")
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)

    # Step 3: Download the model (shows progress, skipped if already cached)
    embedding_model.download(
        lambda progress: print(
            f"\r  [>>] Downloading: {progress:.1f}%", end="", flush=True
        )
    )
    print()  # New line after download progress

    # Step 4: Load the model into memory
    print("  [..] Loading model into memory...")
    embedding_model.load()
    print("  [OK] Embedding model ready!")

    # Step 5: Get the embedding client (used to generate vectors)
    embedding_client = embedding_model.get_embedding_client()

    return embedding_client, embedding_model


def generate_embeddings(chunks: list[dict], embedding_client) -> list[dict]:
    """
    Generate embedding vectors for all text chunks.

    Each chunk's text is converted into a numerical vector (a list
    of floating-point numbers) by the embedding model. These vectors
    capture the semantic meaning of the text.

    We process chunks in batches for efficiency — sending multiple
    texts to the model at once is faster than one at a time.

    Args:
        chunks:           List of chunk dicts (each has a 'text' key)
        embedding_client: The Foundry Local embedding client

    Returns:
        The same list of chunk dicts, now with an 'embedding' key
        added to each, containing the vector as a list of floats.
    """
    print(f"\n[VEC] Generating embeddings for {len(chunks)} chunks...")

    # Extract just the text from each chunk for batch embedding
    texts = [chunk["text"] for chunk in chunks]

    # Batch size for embedding requests
    # Smaller batches use less memory; larger batches are faster
    BATCH_SIZE = 16
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Call the Foundry Local SDK to generate embeddings
        # This runs the model entirely on your device!
        response = embedding_client.generate_embeddings(batch)

        # Extract the embedding vectors from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # Attach each embedding vector to its corresponding chunk
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    # Report the embedding dimensions (useful for debugging)
    if all_embeddings:
        vector_dim = len(all_embeddings[0])
        print(f"\n  [OK] All embeddings generated! (vector dimension: {vector_dim})")

    return chunks


# ──────────────────────────────────────────────────────────
# STEP 4: SQLite DATABASE STORAGE
# ──────────────────────────────────────────────────────────
# We store the chunks and their embeddings in a SQLite
# database — a lightweight, serverless database that lives
# in a single file. No database server needed!
#
# Table schema:
#   chunks (
#       id          INTEGER PRIMARY KEY    — unique ID per chunk
#       source      TEXT                   — original filename
#       chunk_index INTEGER                — position within document
#       text        TEXT                   — the actual text content
#       embedding   TEXT                   — JSON-serialized vector
#   )
# ──────────────────────────────────────────────────────────

def create_database(db_path: Path) -> sqlite3.Connection:
    """
    Create (or reset) the SQLite database and define the schema.

    If the database already exists, we drop the old table and
    recreate it. This ensures a clean state on every ingestion run.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        An open SQLite connection ready for inserts
    """
    print(f"\n[DB] Setting up SQLite database at: {db_path.name}")

    # Connect to the database file (creates it if it doesn't exist)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Drop the old table if it exists (clean re-ingestion)
    cursor.execute("DROP TABLE IF EXISTS chunks")

    # Create the chunks table with our schema
    cursor.execute("""
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT    NOT NULL,
            embedding   TEXT    NOT NULL
        )
    """)

    conn.commit()
    print("  [OK] Database schema created (table: 'chunks')")
    return conn


def save_chunks_to_db(conn: sqlite3.Connection, chunks: list[dict]):
    """
    Insert all chunks (with embeddings) into the SQLite database.

    The embedding vectors are serialized as JSON strings before
    storage. When we retrieve them later in the RAG pipeline,
    we'll deserialize them back into Python lists for similarity
    calculations.

    Args:
        conn:   Open SQLite connection
        chunks: List of chunk dicts with 'source', 'chunk_index',
                'text', and 'embedding' keys
    """
    cursor = conn.cursor()

    print(f"\n[SAVE] Inserting {len(chunks)} chunks into the database...")

    for chunk in chunks:
        # Serialize the embedding vector (list of floats) to a JSON string
        # Example: [0.123, -0.456, 0.789, ...] → "[0.123, -0.456, 0.789, ...]"
        embedding_json = json.dumps(chunk["embedding"])

        cursor.execute(
            """
            INSERT INTO chunks (source, chunk_index, text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (chunk["source"], chunk["chunk_index"], chunk["text"], embedding_json)
        )

    conn.commit()
    print(f"  [OK] All {len(chunks)} chunks saved successfully!")


def print_database_summary(conn: sqlite3.Connection):
    """
    Print a summary of what's in the database after ingestion.

    This gives a quick overview of the knowledge base:
    how many chunks per document and total stats.
    """
    cursor = conn.cursor()

    # Total chunks
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total = cursor.fetchone()[0]

    # Chunks per source document
    cursor.execute("""
        SELECT source, COUNT(*) as chunk_count, 
               MIN(LENGTH(text)) as min_len, 
               MAX(LENGTH(text)) as max_len
        FROM chunks 
        GROUP BY source 
        ORDER BY source
    """)
    rows = cursor.fetchall()

    print("\n" + "=" * 55)
    print("KNOWLEDGE BASE SUMMARY")
    print("=" * 55)
    print(f"  Total chunks in database:  {total}")
    print(f"  Source documents:           {len(rows)}")
    print("-" * 55)
    print(f"  {'Document':<30} {'Chunks':>8} {'Size Range':>14}")
    print("-" * 55)
    for source, count, min_len, max_len in rows:
        print(f"  {source:<30} {count:>8} {min_len:>5}-{max_len:>5} chars")
    print("=" * 55)


# ──────────────────────────────────────────────────────────
# MAIN FUNCTION — Orchestrates the entire ingestion pipeline
# ──────────────────────────────────────────────────────────

def main():
    """
    Main entry point for the data ingestion pipeline.

    Orchestrates the full flow:
      1. Ensure directories exist
      2. Read documents from the data folder
      3. Chunk documents into smaller passages
      4. Initialize the embedding model
      5. Generate embeddings for all chunks
      6. Save everything to SQLite
      7. Print a summary and clean up

    This script should be run once (or whenever documents change)
    BEFORE starting the Streamlit app.
    """
    print("=" * 55)
    print("FOUNDRY RAG ASSISTANT -- Data Ingestion Pipeline")
    print("=" * 55)

    start_time = time.time()

    # ── Step 0: Ensure required directories exist ──────────
    ensure_directories()

    # ── Step 1: Read all documents from the data folder ────
    documents = load_documents(DOCS_DIR)

    # ── Step 2: Chunk documents into smaller passages ──────
    chunks = chunk_documents(documents)

    # ── Step 3: Initialize the Foundry Local embedding model
    embedding_client, embedding_model = initialize_embedding_model()

    try:
        # ── Step 4: Generate embedding vectors for all chunks ─
        chunks = generate_embeddings(chunks, embedding_client)

        # ── Step 5: Create the SQLite database and save ───────
        conn = create_database(DB_PATH)
        save_chunks_to_db(conn, chunks)

        # ── Step 6: Print a summary of the knowledge base ─────
        print_database_summary(conn)
        conn.close()

    finally:
        # ── Step 7: Unload the model to free memory ───────────
        print("\n[CLEANUP] Unloading embedding model to free memory...")
        embedding_model.unload()

    # ── Done! ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n[DONE] Ingestion complete in {elapsed:.1f} seconds!")
    print(f"   Database saved to: {DB_PATH}")
    print(f"   Ready for queries. Run 'streamlit run app.py' next.\n")


# ──────────────────────────────────────────────────────────
# Script entry point — runs when you execute: python ingest.py
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
