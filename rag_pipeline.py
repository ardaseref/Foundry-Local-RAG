"""
Core RAG Logic
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

# DATA CLASSES
# We use dataclasses to structure our return values cleanly.
# This makes the code more readable and type-safe compared
# to returning raw dictionaries or tuples.

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

# MATH UTILITIES -- Cosine Similarity
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

# RAG PIPELINE CLASS
# This class encapsulates the entire RAG workflow:
#   - Model initialization (embedding + chat)
#   - Document retrieval from SQLite
#   - Answer generation with strict prompting
#
# Using a class allows the Streamlit app to initialize
# models ONCE and reuse them across multiple user queries,
# which is much faster than reloading for every question.

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

        # MODEL INITIALIZATION
    
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

        # Step 1: Initialize the SDK (singleton-safe)
        print("[AI] Initializing Foundry Local SDK...")
        if FoundryLocalManager.instance is None:
            config = Configuration(app_name=APP_NAME)
            FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

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
        if not self._chat_model:
            # Fallback: search by exact ID
            self._chat_model = next((m for m in manager.catalog.list_models() if CHAT_MODEL in m.id), None)
            
        if not self._chat_model:
            raise ValueError(f"Model '{CHAT_MODEL}' not found in Foundry Local catalog!")

        # ---------------------------------------------------------
        # GPU FIX: Dynamically determine the exact variant ID required
        # for hardware acceleration. explicitly prefer CUDA over generic-gpu.
        # ---------------------------------------------------------
        gpu_variant = None
        available_ids = [v.id for v in self._chat_model.variants]
        print(f"  [DB] Available variants in catalog: {available_ids}")
        
        # Priority 1: CUDA
        for variant in self._chat_model.variants:
            v_id_lower = variant.id.lower()
            if "cuda" in v_id_lower:
                gpu_variant = variant
                break
                
        # Priority 2: Any non-CPU (like generic-gpu, directml)
        if not gpu_variant:
            for variant in self._chat_model.variants:
                if "generic-cpu" not in variant.id.lower():
                    gpu_variant = variant
                    break
                
        if gpu_variant:
            print(f"  [>>] Forcing hardware-accelerated variant: {gpu_variant.id}")
            self._chat_model.select_variant(gpu_variant)
        else:
            print("  [WARN] GPU variant not found in catalog. Using default.")

        # Step 3.5: Explicitly download & register GPU Execution Providers
        # AFTER variant selection but BEFORE loading.
        # This ensures the required EP (e.g., CUDA) is activated for the variant.
        print("[>>] Registering hardware execution providers (GPU/NPU)...")
        try:
            ep_result = manager.download_and_register_eps()
            print(f"  [OK] EP registration result: {ep_result}")
        except Exception as ep_err:
            print(f"  [WARN] EP registration failed ({ep_err}), falling back to defaults.")
            ep_result = None

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
        settings_kwargs = {
            "temperature": CHAT_TEMPERATURE,
            "max_tokens": CHAT_MAX_TOKENS,
        }
        
        # Passing 0.0 to the underlying ONNX runtime causes a silent failure 
        # (returns an empty stream) in some SDK versions.
        if CHAT_FREQUENCY_PENALTY != 0.0:
            settings_kwargs["frequency_penalty"] = CHAT_FREQUENCY_PENALTY
            
        self._chat_client.settings = ChatClientSettings(**settings_kwargs)
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

        # RETRIEVAL: Find relevant chunks from SQLite
    
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
        q_lower = query.lower()
        
        for chunk in db_chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])

            # Hybrid Search Boost: If the query contains the document filename,
            # we heavily boost the similarity score to guarantee retrieval.
            s_lower = chunk["source"].lower()
            s_name_only = s_lower.replace(".txt", "").replace(".md", "")
            
            if s_name_only in q_lower or s_lower in q_lower:
                score += 0.20

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

        # AUGMENTATION: Build the prompt with context
    
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

    def _build_messages(self, query: str, context: str, chat_history: list = None) -> list[dict]:
        """
        Construct the message list for the chat API.

        This implements our two-prompt architecture:
          - NO context → SYSTEM_PROMPT_CHAT  (zero RAG vocabulary)
          - HAS context → SYSTEM_PROMPT_RAG  (grounded answering)

        Args:
            query:   The user's question
            context: The formatted context string from retrieved chunks
            chat_history: Optional list of previous message dicts

        Returns:
            A list of message dicts ready for the chat API
        """
        if context and context.strip():
            # RAG path: inject retrieved documents into the grounded prompt
            system_prompt = SYSTEM_PROMPT_RAG.format(context=context)
            # RAG mode: include up to 1 turn (last 2 messages) of history
            if chat_history:
                history_to_use = chat_history[-2:]
            else:
                history_to_use = []
        else:
            # Casual chat path: clean prompt with zero RAG vocabulary
            system_prompt = SYSTEM_PROMPT_CHAT
            # Chat mode: Limit history to 10 messages (5 turns) to ensure 
            # lightning-fast ONNX inference during the live demo.
            if chat_history:
                history_to_use = chat_history[-10:]
            else:
                history_to_use = []

        messages = [{"role": "system", "content": system_prompt}]
        
        if history_to_use:
            messages.extend(history_to_use)
            
        # Ensure the current query is the final user message if not already included
        if not history_to_use or history_to_use[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        return messages

    def _safe_stream(self, raw_stream, is_chat_mode: bool = False):
        """
        Programmatic Safety Net: Intercepts the streaming response and forcefully
        stops generation if it detects prompt bleed, degenerate loops, or
        vocabulary stagnation.
        """
        import re
        buffer = ""
        in_think_block = False
        think_block_ended = False
        
        # Phrases from our system prompts — if the model outputs these,
        # it's regurgitating its own instructions (prompt bleed).
        forbidden_phrases = [
            "factual answering assistant",
            "use only facts stated",
            "do not add outside knowledge",
            "respond exactly:",
            "helpful ai assistant",
            "1-3 sentences maximum",
            "respond in the same language",
        ]
        
        for chunk in raw_stream:
            if chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta'):
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    
                    # --- Reasoning Tag Filtering Logic ---
                    if "<think>" in buffer and not in_think_block:
                        in_think_block = True
                        
                    if in_think_block and "</think>" in buffer:
                        in_think_block = False
                        # We just ended the block. Extract the remainder.
                        after_think = buffer.split("</think>", 1)[-1].strip()
                        # Reset the buffer to just the text after the think block
                        buffer = after_think
                        # Yield the remaining text that was after the block immediately
                        if after_think:
                            yield after_think
                        continue
                        
                    if in_think_block:
                        # We are inside the reasoning block. Don't yield, don't run guards.
                        continue
                    # -------------------------------------
                    
                    lower_buf = buffer.lower()
                    
                    # Prompt Bleed Detection
                    if any(phrase in lower_buf for phrase in forbidden_phrases):
                        break
                        
                    # Chat Mode Length Guard
                    if is_chat_mode:
                        words = buffer.split()
                        if len(words) > 300:
                            yield "..."
                            break

                    # Loop detection removed per user request: We will let the model generate naturally, 
                    # even if it loops on creative edge-cases like poetry, because hard-stops look worse.
                    
                    # If we got here, we are not in a think block, and we haven't just exited one.
                    # Yield the new raw chunk directly. But guard against cases where the raw chunk
                    # contains fragments of the think tags that didn't trigger the state changes above.
                    clean_content = content.replace("<think>", "").replace("</think>", "")
                    if clean_content and not "<think>" in clean_content:
                        yield clean_content

        # GENERATION: Get answers from the local LLM
    
    def query(self, question: str, top_k: int = TOP_K, chat_history: list = None) -> RAGResponse:
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
        is_chat_mode = False
        if relevant_chunks:
            context = self._build_context_string(relevant_chunks)
        else:
            # Empty context triggers the "chat normally" system prompt behavior
            context = ""
            is_chat_mode = True

        messages = self._build_messages(question, context, chat_history=chat_history)
        
        # Generation settings are already globally defined in initialize()

        # ── GENERATE: Get answer from local LLM ──────────
        # Use streaming to collect the full response token-by-token
        answer_parts = []
        try:
            raw_stream = self._chat_client.complete_streaming_chat(messages)
            for content in self._safe_stream(raw_stream, is_chat_mode=is_chat_mode):
                answer_parts.append(content)
        except Exception as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "illegal memory" in error_msg:
                answer_parts.append("\n\n⚠️ GPU error occurred. Please refresh the page to recover.")
            else:
                answer_parts.append(f"\n\n⚠️ Error: {error_msg}")

        full_answer = "".join(answer_parts)

        return RAGResponse(
            answer=full_answer,
            sources=relevant_chunks,
            query=question
        )

    def query_stream(self, question: str, top_k: int = TOP_K, chat_history: list = None):
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
            chat_history: Optional list of previous message dicts

        Yields:
            Tuples of (token: str | None, sources: list[RetrievedChunk])
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # ── RETRIEVE: Find relevant document chunks ───────
        relevant_chunks = self.get_top_chunks(question, top_k=top_k)

        # ── AUGMENT: Build the prompt with context ────────
        is_chat_mode = False
        if relevant_chunks:
            context = self._build_context_string(relevant_chunks)
        else:
            # Empty context triggers the "chat normally" system prompt behavior
            context = ""
            is_chat_mode = True

        messages = self._build_messages(question, context, chat_history=chat_history)
        
        # Generation settings are already globally defined in initialize()

        # ── GENERATE: Stream answer tokens from local LLM ─
        first_token = True
        try:
            raw_stream = self._chat_client.complete_streaming_chat(messages)
            for content in self._safe_stream(raw_stream, is_chat_mode=is_chat_mode):
                if first_token:
                    yield content, relevant_chunks
                    first_token = False
                else:
                    yield content, []
        except Exception as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "illegal memory" in error_msg:
                yield "\n\n⚠️ GPU error occurred. Please refresh the page to recover.", relevant_chunks if first_token else []
            else:
                yield f"\n\n⚠️ Error: {error_msg}", relevant_chunks if first_token else []

        # UTILITY METHODS
    
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

# STANDALONE TEST MODE
# When run directly (python rag_pipeline.py), this script
# performs a quick end-to-end test of the RAG pipeline.
# This is useful for debugging before launching Streamlit.

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
