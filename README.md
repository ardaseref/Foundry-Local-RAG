# Enterprise Foundry RAG Assistant

A 100% local, privacy-first Retrieval-Augmented Generation (RAG) pipeline built with the **Microsoft Foundry Local SDK**, **Streamlit**, and **SQLite**. 

This architecture leverages the 3.8B parameter `phi-4-mini` model for highly factual reasoning and the `qwen3-embedding` model for perfect cross-lingual semantic search—all running entirely on local hardware (zero cloud API dependencies).

## 🚀 Key Features

- **100% Offline & Private:** Your documents and chat data never leave your machine.
- **Cross-Lingual Intelligence:** Ask questions in Turkish about English documents (or vice-versa) with perfect factual retrieval.
- **Precision Sentence-Boundary Chunking:** A custom ingestion engine that respects sentence integrity, ensuring no words are sliced in half during embedding.
- **Dual-Prompt Anti-Hallucination:** Dynamically swaps between "Casual Chat" and "Strict RAG" modes to prevent the model from inventing facts when answering general queries.
- **Cinematic UI:** A buttery-smooth, stutter-free Streamlit interface featuring typewriter streaming and rolling memory management to prevent VRAM overflow.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM Engine:** Microsoft Foundry Local SDK (ONNX runtime)
- **Chat Model:** `phi-4-mini` (3.8B parameters)
- **Embedding Model:** `qwen3-embedding-0.6b`
- **Vector Database:** SQLite + local cosine similarity math

## 📥 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ardaseref/Foundry-Local-RAG.git
   cd Foundry-Local-RAG
   ```

2. **Activate the Virtual Environment:**
   *(Ensure you have your environment setup with the required dependencies)*
   ```bash
   .\venv\Scripts\activate
   ```

3. **Ingest the Knowledge Base:**
   Place any `.txt` files you want the AI to learn into the `data/sample_docs/` folder, then run the ingestion script:
   ```bash
   python ingest.py
   ```
   *This will chunk, embed, and store your documents into the local SQLite database.*

4. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Architecture Highlights
- **Context Window Protection:** The UI intelligently buffers the rolling chat history to prevent the strict local GPU context limits from overflowing during long sessions.
- **Token Output Constraint:** Generation is carefully capped to ensure rapid, punchy responses ideal for live enterprise demonstrations.
