# GenAI Agent System

## Overview

GenAI Agent System is a CLI-based AI assistant that enables users to ingest documents, perform retrieval-based question answering, execute tools, and maintain persistent conversational memory.

The system follows a modular architecture with abstraction layers for LLMs, embeddings, vector databases, memory systems, and tools, making it easy to extend and maintain.

---

## Features

### Document Ingestion

* PDF support
* DOCX support
* CSV support
* Excel support
* Text extraction
* Table extraction
* Semantic chunking

### Retrieval-Augmented Generation (RAG)

* Embedding generation using Sentence Transformers
* Persistent vector storage using ChromaDB
* Similarity search
* Cross-encoder reranking

### Agent-Based Architecture

* Reasoning Agent
* Summarization Agent
* Tool Agent

### Tool Execution

* Calculator Tool
* Extensible tool framework

### Memory System

* Episodic memory persistence
* SQLite storage
* Conversation history retrieval
* Background summarization

---

## System Workflow

1. User provides a document.
2. The system extracts text and tables.
3. Content is split into semantic chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in ChromaDB.
6. User asks a question.
7. Tool Agent checks whether a tool should be used.
8. If not, Retriever fetches relevant chunks.
9. Reasoning Agent generates an answer using the LLM.
10. Interaction is stored in memory.
11. Background summarization runs periodically.

---

## Project Structure

```text
agents/
embeddings/
indexing/
ingestion/
llm/
memory/
retrieval/
tools/
utils/
vector_db/

main.py
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=your_model

RERANKER_MODEL=your_reranker_model

MEMORY_DB_PATH=memory.db
```

### Run

```bash
python main.py
```

---

## Usage

### Ask document questions

```text
What are the key findings in the report?
```

### Calculator queries

```text
23 * 45 + 100
```

### View summaries

```text
summary
```

### Exit

```text
exit
```

or

```text
bye
```

---

## Technologies Used

* Python
* LangChain Text Splitters
* Sentence Transformers
* ChromaDB
* SQLite
* OpenRouter
* CrossEncoder Reranker

---

## Design Principles

* Modular architecture
* Separation of concerns
* Extensibility through abstraction
* Persistent memory
* Retrieval-Augmented Generation
* Tool-based execution

---
