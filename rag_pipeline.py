"""
============================================================
rag_pipeline.py -- Core RAG (Retrieval-Augmented Generation) Logic
============================================================
This module is the BRAIN of our RAG assistant. It connects:

  1. RETRIEVAL  -- Finds relevant document chunks from SQLite
                   using cosine similarity between vectors
  2. AUGMENTATION -- Builds a prompt with retrieved context
  3. GENERATION -- Sends the augmented prompt to the local LLM
                   (via Microsoft Foundry Local) for answering

This module is designed to be IMPORTED by our Streamlit frontend.
It exposes clean functions that the UI can call directly:

    from rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.initialize()
    answer, sources = pipeline.query("What is Foundry Local?")

Author:  Enterprise RAG Assistant Project
============================================================
"""

import json
import math
import sqlite3
from dataclasses import dataclass, field

# Foundry Local SDK -- Microsoft's on-device AI runtime
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.openai.chat_client import ChatClientSettings

# Import our centralized configuration
from config import (
    APP_NAME,
    CHAT_FREQUENCY_PENALTY,
    CHAT_MAX_TOKENS,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    DB_PATH,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_RAG,
    TOP_K,
)


# ──────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────
# We use dataclasses to structure our return values cleanly.
# This makes the code more readable and type-safe compared
# to returning raw dictionaries or tuples.
# ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    Represents a single chunk retrieved from the knowledge base.

    Attributes:
        chunk_id:    Unique database ID of this chunk
        source:      Original document filename (e.g., "foundry_local.txt")
        chunk_index: Position within the source document (0-based)
        text:        The actual text content of the chunk
        score:       Cosine similarity score (0.0 to 1.0, higher = more relevant)
    """
    chunk_id: int
    source: str
    chunk_index: int
    text: str
    score: float


@dataclass
class RAGResponse:
    """
    The complete response from a RAG query.

    Attributes:
        answer:  The generated text answer from the LLM
        sources: List of RetrievedChunk objects used as context
        query:   The original user question
    """
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""


# ──────────────────────────────────────────────────────────
# MATH UTILITIES -- Cosine Similarity
# ──────────────────────────────────────────────────────────
# Cosine similarity measures how "aligned" two vectors are.
# It ranges from -1 (opposite) to +1 (identical direction).
#
# For text embeddings:
#   - Score near 1.0 = very similar meaning
#   - Score near 0.0 = unrelated topics
#   - Score < 0.0    = opposite meaning (rare in practice)
#
# We compute this in pure Python (no numpy needed) to keep
# dependencies minimal and the code easy to understand.
# ──────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula:
        similarity = (A . B) / (||A|| * ||B||)

    Where:
        A . B   = dot product (sum of element-wise multiplication)
        ||A||   = magnitude of A (square root of sum of squares)

    Args:
        vec_a: First embedding vector (list of floats)
        vec_b: Second embedding vector (list of floats)

    Returns:
        A float between -1.0 and 1.0 (typically 0.0 to 1.0 for embeddings)
    """
    # Calculate the dot product: sum of (a_i * b_i) for all dimensions
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Calculate the magnitude (L2 norm) of each vector
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # Avoid division by zero (shouldn't happen with real embeddings)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ──────────────────────────────────────────────────────────
# RAG PIPELINE CLASS
# ──────────────────────────────────────────────────────────
# This class encapsulates the entire RAG workflow:
#   - Model initialization (embedding + chat)
#   - Document retrieval from SQLite
#   - Answer generation with strict prompting
#
# Using a class allows the Streamlit app to initialize
# models ONCE and reuse them across multiple user queries,
# which is much faster than reloading for every question.
# ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Enterprise RAG Pipeline using Microsoft Foundry Local.

    This class manages the full lifecycle of a RAG query:
      1. Initialize models (done once at startup)
      2. Embed user queries
      3. Retrieve relevant chunks from SQLite
      4. Generate grounded answers via local LLM

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()                  # Load models (once)
        response = pipeline.query("question")  # Ask questions (many times)
        pipeline.shutdown()                    # Free resources (at exit)
    """

    def __init__(self):
        """Initialize the pipeline (models are NOT loaded yet)."""
        self._embedding_client = None
        self._chat_client = None
        self._embedding_model = None
        self._chat_model = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized with models."""
        return self._is_initialized

    # ──────────────────────────────────────────────────────
    # MODEL INITIALIZATION
    # ──────────────────────────────────────────────────────

    def initialize(self, progress_callback=None):
        """
        Initialize the Foundry Local SDK and load both models.

        This downloads (if needed) and loads:
          - The EMBEDDING model (for converting text to vectors)
          - The CHAT model (for generating answers)

        This should be called ONCE when the application starts.
        Subsequent calls are no-ops if already initialized.

        Args:
            progress_callback: Optional function(stage: str, progress: float)
                               called during download/load for UI updates.
                               stage is one of: "embedding_download", "embedding_load",
                               "chat_download", "chat_load"
        """
        if self._is_initialized:
            return  # Already initialized, skip

        def _notify(stage, value=None):
            if progress_callback:
                progress_callback(stage, value)

        # Step 1: Initialize the SDK
        print("[AI] Initializing Foundry Local SDK...")
        config = Configuration(app_name=APP_NAME)
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

        # Step 1.5: Explicitly download & register GPU Execution Providers.
        # Without this, the SDK may silently fall back to CPU.
        # This forces it to discover and activate DirectML / CUDA / etc.
        print("[>>] Registering hardware execution providers (GPU/NPU)...")
        try:
            ep_result = manager.download_and_register_eps()
            print(f"  [OK] EP registration result: {ep_result}")
        except Exception as ep_err:
            print(f"  [WARN] EP registration failed ({ep_err}), falling back to defaults.")
            ep_result = None

        # Step 2: Load the EMBEDDING model
        print(f"[>>] Loading embedding model: {EMBEDDING_MODEL}")
        self._embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
        self._embedding_model.download(
            lambda p: (_notify("embedding_download", p),
                       print(f"\r  [>>] Downloading embedding model: {p:.1f}%", end="", flush=True))
        )
        print()
        _notify("embedding_load")
        self._embedding_model.load()
        self._embedding_client = self._embedding_model.get_embedding_client()
        print("  [OK] Embedding model ready!")

        # Step 3: Load the CHAT model
        print(f"[>>] Loading chat model: {CHAT_MODEL}")
        self._chat_model = manager.catalog.get_model(CHAT_MODEL)

        # ---------------------------------------------------------
        # GPU FIX: Dynamically determine the exact variant ID required
        # for hardware acceleration instead of guessing strings.
        # ---------------------------------------------------------
        gpu_variant = None
        available_ids = [v.id for v in self._chat_model.variants]
        print(f"  [DB] Available variants in catalog: {available_ids}")
        
        for variant in self._chat_model.variants:
            v_id_lower = variant.id.lower()
            # Any variant that isn't the generic CPU fallback is hardware-accelerated
            if "generic-cpu" not in v_id_lower:
                gpu_variant = variant
                break
                
        if gpu_variant:
            print(f"  [>>] Forcing hardware-accelerated variant: {gpu_variant.id}")
            self._chat_model.select_variant(gpu_variant)
        else:
            print("  [WARN] GPU variant not found in catalog. Using default.")

        self._chat_model.download(
            lambda p: (_notify("chat_download", p),
                       print(f"\r  [>>] Downloading chat model: {p:.1f}%", end="", flush=True))
        )
        print()
        _notify("chat_load")
        self._chat_model.load()
        self._chat_client = self._chat_model.get_chat_client()
        
        # ── Configure deterministic generation parameters ─
        # This is applied at the client level, so every call to
        # complete_chat / complete_streaming_chat automatically
        # uses these settings — no per-call arguments needed.
        self._chat_client.settings = ChatClientSettings(
            temperature=CHAT_TEMPERATURE,
            max_tokens=CHAT_MAX_TOKENS,
            frequency_penalty=CHAT_FREQUENCY_PENALTY,
        )
        print(f"  [OK] Chat model ready! "
              f"(temp={CHAT_TEMPERATURE}, max_tokens={CHAT_MAX_TOKENS}, "
              f"freq_penalty={CHAT_FREQUENCY_PENALTY})")

        self._is_initialized = True

        # Step 4: Log detected hardware acceleration
        self._log_hardware_info(ep_result)

        print("[OK] RAG Pipeline fully initialized!\n")

    def _log_hardware_info(self, ep_result=None):
        """
        Log which hardware acceleration backend was registered.

        Args:
            ep_result: The result from download_and_register_eps(), if available.
        """
        try:
            if ep_result is not None:
                print(f"  [HW] Execution Providers registered: {ep_result}")
            else:
                # Fallback: attempt to read EP info from the model
                ep = getattr(self._chat_model, 'execution_provider', None)
                if ep:
                    print(f"  [HW] Execution Provider: {ep}")
                else:
                    model_info = getattr(self._chat_model, 'info', None)
                    if model_info:
                        variant = getattr(model_info, 'variant', None)
                        if variant and 'cuda' in str(variant).lower():
                            print("  [HW] Detected: CUDA (NVIDIA GPU)")
                        elif variant and 'dml' in str(variant).lower():
                            print("  [HW] Detected: DirectML (GPU)")
                        else:
                            print(f"  [HW] Model variant: {variant or 'auto-selected'}")
                    else:
                        print("  [HW] Execution Provider: auto-detected by SDK")
        except Exception:
            print("  [HW] Execution Provider: could not determine")

    # ──────────────────────────────────────────────────────
    # RETRIEVAL: Find relevant chunks from SQLite
    # ──────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """
        Convert a user's question into an embedding vector.

        This uses the SAME embedding model that was used during
        ingestion, ensuring the vectors are in the same "space"
        and can be compared with cosine similarity.

        Args:
            query: The user's question text

        Returns:
            A list of floats representing the query's embedding vector
        """
        response = self._embedding_client.generate_embedding(query)
        return response.data[0].embedding

    def _fetch_all_chunks_from_db(self) -> list[dict]:
        """
        Load all chunks and their embeddings from the SQLite database.

        For our small knowledge base (< 100 chunks), loading everything
        into memory is fast and simple. For larger datasets (10,000+
        chunks), you would use a dedicated vector database instead.

        Returns:
            A list of dicts with keys: id, source, chunk_index, text, embedding
        """
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, source, chunk_index, text, embedding
            FROM chunks
        """)

        rows = cursor.fetchall()
        conn.close()

        chunks = []
        for row in rows:
            chunks.append({
                "id": row[0],
                "source": row[1],
                "chunk_index": row[2],
                "text": row[3],
                # Deserialize the JSON-encoded embedding back into a Python list
                "embedding": json.loads(row[4])
            })

        return chunks

    def get_top_chunks(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a user query.

        This is the RETRIEVAL step of RAG. It works as follows:

          1. Embed the user's query (convert text -> vector)
          2. Load all stored chunks from SQLite
          3. Compute cosine similarity between query and each chunk
          4. Sort by similarity score (highest first)
          5. Return the top-K most relevant chunks

        Args:
            query: The user's question text
            top_k: Number of top chunks to return (default from config)

        Returns:
            A list of RetrievedChunk objects, sorted by relevance (best first)
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Step 1: Convert the query text into an embedding vector
        query_embedding = self._embed_query(query)

        # Step 2: Load all chunks from the database
        db_chunks = self._fetch_all_chunks_from_db()

        if not db_chunks:
            print("[WARN] No chunks found in the database. Run ingest.py first.")
            return []

        # Step 3: Calculate cosine similarity for EVERY chunk
        scored_chunks = []
        for chunk in db_chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])

            # Only keep chunks above the minimum similarity threshold
            # This filters out completely irrelevant results
            if score >= SIMILARITY_THRESHOLD:
                scored_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk["id"],
                        source=chunk["source"],
                        chunk_index=chunk["chunk_index"],
                        text=chunk["text"],
                        score=score
                    )
                )

        # Step 4: Sort by score (highest similarity first)
        scored_chunks.sort(key=lambda c: c.score, reverse=True)

        # Step 5: Return only the top-K results
        return scored_chunks[:top_k]

    # ──────────────────────────────────────────────────────
    # AUGMENTATION: Build the prompt with context
    # ──────────────────────────────────────────────────────

    def _build_context_string(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Each chunk is labeled with its source document name so the
        model can cite sources in its answer. The format is:

            [Document: foundry_local.txt]
            The actual text content of this chunk...

            [Document: rag_overview.txt]
            The actual text content of this chunk...

        Args:
            chunks: List of RetrievedChunk objects to include

        Returns:
            A formatted string containing all chunk texts with source labels
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Use plain "Source:" labels instead of bracket tags.
            # Bracket-style tags like [Document: ...] cause small models
            # to hallucinate more tags in a loop.
            context_parts.append(
                f"Source {i} ({chunk.source}):\n{chunk.text}"
            )
        return "\n\n".join(context_parts)

    def _build_messages(self, query: str, context: str) -> list[dict]:
        """
        Construct the chat messages array for the LLM.

        Uses a TWO-PROMPT architecture instead of a single template
        with If/Else logic:
          - NO context → SYSTEM_PROMPT_CHAT  (zero RAG vocabulary)
          - HAS context → SYSTEM_PROMPT_RAG  (grounded answering)

        This prevents the small model from seeing RAG-related words
        ("sources", "context", "documents") during casual chat,
        which was the primary cause of hallucinated source citations.

        The system prompt uses our strict enterprise template from
        config.py, which enforces:
          - Answer ONLY from context
          - Cite sources
          - Say "I don't know" when info is missing

        Args:
            query:   The user's question
            context: The formatted context string from retrieved chunks

        Returns:
            A list of message dicts ready for the chat API
        """
        if context and context.strip():
            # RAG path: inject retrieved documents into the grounded prompt
            system_prompt = SYSTEM_PROMPT_RAG.format(context=context)
        else:
            # Casual chat path: clean prompt with zero RAG vocabulary
            system_prompt = SYSTEM_PROMPT_CHAT

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

    # ──────────────────────────────────────────────────────
    # GENERATION: Get answers from the local LLM
    # ──────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = TOP_K) -> RAGResponse:
        """
        Execute a full RAG query: Retrieve -> Augment -> Generate.

        This is the main entry point for asking questions. It:
          1. Retrieves relevant chunks from the knowledge base
          2. Builds an augmented prompt with context and citations
          3. Sends the prompt to the local LLM for answer generation
          4. Returns the answer along with source references

        If no relevant chunks are found, the model is instructed
        to explicitly say it doesn't have the information.

        Args:
            question: The user's question text
            top_k:    Number of context chunks to retrieve (default from config)

        Returns:
            A RAGResponse object containing:
              - answer:  The generated text answer
              - sources: The retrieved chunks used as context
              - query:   The original question
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # ── RETRIEVE: Find relevant document chunks ───────
        relevant_chunks = self.get_top_chunks(question, top_k=top_k)

        # ── AUGMENT: Build the prompt with context ────────
        if relevant_chunks:
            context = self._build_context_string(relevant_chunks)
        else:
            # No chunks passed the similarity threshold.
            # Pass EMPTY context so the system prompt's "chat normally" rule activates.
            context = ""

        messages = self._build_messages(question, context)

        # ── GENERATE: Get answer from local LLM ──────────
        # Use streaming to collect the full response token-by-token
        answer_parts = []
        for chunk in self._chat_client.complete_streaming_chat(messages):
            if chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta'):
                content = chunk.choices[0].delta.content
                if content:
                    answer_parts.append(content)

        full_answer = "".join(answer_parts)

        return RAGResponse(
            answer=full_answer,
            sources=relevant_chunks,
            query=question
        )

    def query_stream(self, question: str, top_k: int = TOP_K):
        """
        Execute a RAG query with STREAMING response.

        This is the preferred method for the Streamlit UI because
        it yields answer tokens one at a time, creating a smooth
        "typing" effect as the model generates its response.

        The method yields tuples of (token, sources) where:
          - token:   A small piece of the answer text (str or None)
          - sources: The full list of retrieved chunks (sent with first token)

        Usage in Streamlit:
            for token, sources in pipeline.query_stream("question"):
                if token:
                    display_token(token)

        Args:
            question: The user's question text
            top_k:    Number of context chunks to retrieve

        Yields:
            Tuples of (token: str | None, sources: list[RetrievedChunk])
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # ── RETRIEVE: Find relevant document chunks ───────
        relevant_chunks = self.get_top_chunks(question, top_k=top_k)

        # ── AUGMENT: Build the prompt with context ────────
        if relevant_chunks:
            context = self._build_context_string(relevant_chunks)
        else:
            # Empty context triggers the "chat normally" system prompt behavior
            context = ""

        messages = self._build_messages(question, context)

        # ── GENERATE: Stream answer tokens from local LLM ─
        first_token = True
        for chunk in self._chat_client.complete_streaming_chat(messages):
            if chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta'):
                content = chunk.choices[0].delta.content
                if content:
                    if first_token:
                        # Send sources with the first token
                        yield content, relevant_chunks
                        first_token = False
                    else:
                        yield content, []

    # ──────────────────────────────────────────────────────
    # UTILITY METHODS
    # ──────────────────────────────────────────────────────

    def get_knowledge_base_stats(self) -> dict:
        """
        Get statistics about the current knowledge base.

        Returns a dict with:
          - total_chunks: Total number of chunks in the database
          - documents:    List of unique source document names
          - doc_count:    Number of unique source documents
          - db_path:      Path to the SQLite database file

        This is useful for displaying KB info in the Streamlit sidebar.
        """
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            # Total chunk count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            # Unique source documents with chunk counts
            cursor.execute("""
                SELECT source, COUNT(*) as chunk_count
                FROM chunks
                GROUP BY source
                ORDER BY source
            """)
            doc_stats = cursor.fetchall()
            conn.close()

            return {
                "total_chunks": total_chunks,
                "documents": [row[0] for row in doc_stats],
                "doc_stats": {row[0]: row[1] for row in doc_stats},
                "doc_count": len(doc_stats),
                "db_path": str(DB_PATH),
            }
        except Exception:
            return {
                "total_chunks": 0,
                "documents": [],
                "doc_stats": {},
                "doc_count": 0,
                "db_path": str(DB_PATH),
            }

    def shutdown(self):
        """
        Unload all models and free memory.

        Call this when the application is shutting down to cleanly
        release the model resources. The models can be reloaded
        by calling initialize() again if needed.
        """
        if self._embedding_model:
            print("[CLEANUP] Unloading embedding model...")
            self._embedding_model.unload()
            self._embedding_model = None
            self._embedding_client = None

        if self._chat_model:
            print("[CLEANUP] Unloading chat model...")
            self._chat_model.unload()
            self._chat_model = None
            self._chat_client = None

        self._is_initialized = False
        print("[OK] All models unloaded. Resources freed.")


# ──────────────────────────────────────────────────────────
# STANDALONE TEST MODE
# ──────────────────────────────────────────────────────────
# When run directly (python rag_pipeline.py), this script
# performs a quick end-to-end test of the RAG pipeline.
# This is useful for debugging before launching Streamlit.
# ──────────────────────────────────────────────────────────

def main():
    """
    Interactive test mode for the RAG pipeline.

    Initializes models, then enters a loop where you can type
    questions and see the full RAG response with source citations.
    """
    print("=" * 60)
    print("FOUNDRY RAG ASSISTANT -- Pipeline Test Mode")
    print("=" * 60)

    # Create and initialize the pipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Show knowledge base stats
    stats = pipeline.get_knowledge_base_stats()
    print(f"\nKnowledge Base: {stats['total_chunks']} chunks from {stats['doc_count']} documents")
    for doc, count in stats["doc_stats"].items():
        print(f"  - {doc}: {count} chunks")

    print("\nType your questions below. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("Question: ").strip()
            if not question or question.lower() == "quit":
                break

            # Execute the full RAG query
            response = pipeline.query(question)

            # Display the answer
            print(f"\nAnswer: {response.answer}")

            # Display source citations
            if response.sources:
                print("\nSources:")
                for src in response.sources:
                    print(f"  [{src.source}] (similarity: {src.score:.3f})")
                    print(f"    \"{src.text[:80]}...\"")
            else:
                print("\n  (No relevant sources found)")

            print()

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Shutting down...")

    finally:
        pipeline.shutdown()

    print("Done!")


if __name__ == "__main__":
    main()
