🧠 AI Agent – RAG-based Document Intelligence System

🏆 Runner-up | TCS AI Friday Innovation Challenge

📖 Overview

This project was developed as part of TCS AI Friday, showcasing an AI-powered intelligent agent that leverages Retrieval-Augmented Generation (RAG) to understand and summarize enterprise documents.

The agent can ingest documents, generate vector embeddings, and answer or summarize queries based on contextual data retrieved from a local FAISS vector store.
Our team earned the Runner-up position in the challenge for building a modular, production-ready RAG pipeline integrating TCS GenAI Lab APIs.

⚙️ Core Components
1. data_loader.py

Loads data from multiple formats — PDF, DOCX, TXT, CSV, Excel, JSON — using LangChain’s document loaders.

2. embedding.py

Splits text into manageable chunks and generates text embeddings using TCS GenAI Lab OpenAI-compatible API.

3. vectorstore.py

Builds and maintains a FAISS-based vector database, allowing efficient semantic search and retrieval.

4. search.py

Implements a RAG pipeline combining document retrieval with LLM summarization, using GenAI Lab GPT-4o-mini for contextual responses.
            ┌───────────────────┐
            │   Data Loader     │
            │ (PDF, CSV, DOCX)  │
            └─────────┬─────────┘
                      │
                      ▼
           ┌────────────────────┐
           │  EmbeddingPipeline │
           │ (Chunk + Embed)    │
           └─────────┬──────────┘
                     │
                     ▼
            ┌───────────────────┐
            │   FAISS Vector DB │
            │ (Semantic Search) │
            └─────────┬─────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │    RAG Search Agent  │
           │ (Retrieve + LLM Summ)│
           └──────────────────────┘
           
