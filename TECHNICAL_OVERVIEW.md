# Technical Overview

This document explains the files and modules in the project, including their purpose and responsibilities.

## Root files

- `main.py`
  - Orchestrates the CLI interface
  - Starts document indexing and the user query loop
  - Builds core objects: ingestion pipeline, retrieval, agents, memory
  - Saves interactions and triggers background summarization

## `ingestion/`

- `document_loader.py`
  - Core document ingestion logic
  - Combines text and table extraction
  - Creates text, table, and table-row chunks

- `text_extractor.py`
  - Extracts text from PDF, DOCX, CSV, and Excel
  - Returns raw text for chunking

- `table_extractor.py`
  - Extracts table data from PDFs and spreadsheets
  - Returns dataframes and page metadata

- `chunker.py`
  - Splits long text into smaller semantically meaningful pieces
  - Uses `RecursiveCharacterTextSplitter`

- `chunk_factory.py`
  - Builds chunk dictionaries with `chunk_id` and metadata
  - Generates UUIDs for each chunk from source and content

- `table_summarizer.py`
  - Produces summaries for tables and table rows
  - Includes row previews and unique values in the summary

## `embeddings/`

- `base.py`
  - Abstract base class for embedding providers
  - Defines `embed(text)` interface

- `sentence_transformer_embedding.py`
  - Implements `BaseEmbedding`
  - Uses `SentenceTransformer` to generate normalized vectors

## `vector_db/`

- `base_vector_db.py`
  - Abstract base class for vector databases
  - Defines `upsert(...)` and `query(...)`

- `chroma_db.py`
  - Implements persistent Chroma storage
  - Supports upsert and k-nearest neighbor query
  - Includes document text and metadata in the stored records

## `indexing/`

- `indexing_pipeline.py`
  - Connects ingestion, embedding, and vector persistence
  - Iterates over chunks and upserts vectors and metadata

## `retrieval/`

- `retriever.py`
  - Executes query embedding and vector database search
  - Converts Chroma results into chunk dictionaries
  - Calls reranker for final candidate selection

- `reranker.py`
  - Uses `CrossEncoder` to score query/chunk pairs
  - Returns top ranked chunks by semantic relevance

## `agents/`

- `reasoning_agent.py`
  - Builds the reasoning prompt for the LLM
  - Adds system instructions, conversation history, and retrieved context
  - Produces the final answer text

- `summarization_agent.py`
  - Summarizes conversation snippets
  - Uses a lower-temperature `LLMConfig` for more stable summaries

- `tool_agent.py`
  - Determines if a query should use a tool
  - Current logic recognizes calculator expressions only
  - Dispatches calculator execution when appropriate

## `tools/`

- `base_tool.py`
  - Abstract tool interface
  - Defines `execute(params)`

- `calculator_tool.py`
  - Evaluates arithmetic expressions from the user query
  - Returns a structured output dictionary

- `tool_registry.py`
  - Stores available tool instances
  - Makes tool lookup simple and extensible

## `llm/`

- `base_llm.py`
  - Abstract base class for LLM providers
  - Defines `generate(messages, config)`

- `openrouter_llm.py`
  - OpenRouter implementation of `BaseLLM`
  - Sends chat messages and temperature/top_p/max_tokens settings

- `llm_config.py`
  - Defines runtime configuration for LLM calls
  - Supports `temperature`, `top_p`, and `max_tokens`

## `memory/`

- `base_memory.py`
  - Abstract memory interface
  - Defines methods to save and read interactions and summaries

- `episodic_memory.py`
  - Implements `BaseMemory`
  - Uses `MemoryRepository` for persistence

- `memory_repository.py`
  - SQLite-based persistence layer
  - Stores tables `interactions` and `memory_summaries`
  - Provides retrieval of recent interactions and all summaries

- `memory_manager.py`
  - Orchestrates memory operations
  - Starts background summarization threads
  - Builds chat context from recent interactions

## `utils/`

- `logger.py`
  - Provides logging capabilities used across the app

## Key patterns and design choices

- Abstract base classes enable extensibility for LLM, embedding, vector DB, tool, and memory layers.
- Document ingestion is separated into text and table handling.
- Chunk creation uses deterministic UUIDs so each chunk is consistently identified.
- Retrieval uses vector search followed by reranking for better relevance.
- Memory is persisted externally using SQLite, not just in memory.
- Background summarization runs with a daemon thread to avoid blocking the main loop.
