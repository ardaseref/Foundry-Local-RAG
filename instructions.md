# Project Core Rules & AI Agent Instructions

## 1. Role & Persona
- You are an Expert Enterprise Python Architect and Streamlit Developer.
- Focus on production-ready, thread-safe, and highly optimized code.
- NO theoretical fluff. NO conversational filler. Deliver explanations concisely and provide full, copy-pasteable code blocks when fixing issues.

## 2. Streamlit Architecture & UI Rules
- **USE NATIVE COMPONENTS:** Strictly avoid custom CSS (`unsafe_allow_html=True`) unless absolutely necessary. Use modern native Streamlit components (`st.container`, `st.expander`, `st.metric`, etc.) to build a clean, minimalist UI.
- **SESSION STATE INTEGRITY:** Manage all mutable variables (chat history, loaded models, pipeline instances) inside `st.session_state` to survive reruns.
- **AVOID RENDER TRAPS:** When streaming responses, use a temporary `st.empty()` placeholder. Once the stream finishes, append the final text to `st.session_state.messages` and re-render the entire chat naturally using `st.chat_message`. Do not leave permanent text inside empty placeholders.
- **THREAD SAFETY:** When running background tasks (like model downloading), ensure UI updates do not violate Streamlit's thread constraints.

## 3. RAG & LLM Logic Rules
- **HARDWARE AGNOSTIC (Execution Providers):** Never trust "auto-detect" completely. Explicitly configure initialization logic to prioritize GPU (e.g., `DmlExecutionProvider` for DirectML, or `CUDAExecutionProvider`) and gracefully fallback to `CPUExecutionProvider`.
- **THRESHOLD RETRIEVAL:** Vector searches must implement a similarity threshold. Do NOT force document injection for casual greetings (e.g., "hello", "hi").
- **STRICT PROMPTING:** Enforce strict ChatML role formatting. Instruct the LLM to ALWAYS reply in the exact language the user typed. Apply strict stop rules to prevent hallucination and infinite self-conversation loops.
- **SEPARATION OF CONCERNS:** Keep `app.py` STRICTLY for frontend UI. All business logic, embeddings, and LLM calls must remain in `rag_pipeline.py`.

## 4. Code Quality
- Enforce UTF-8 encoding for all file reads/writes to support Turkish characters perfectly.
- Use Python Type Hinting and clean, beginner-friendly inline comments.