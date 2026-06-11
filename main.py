import os
from dotenv import load_dotenv

from config.runtime_state import RuntimeState
from agents.reasoning_agent import ReasoningAgent
from agents.summarization_agent import SummarizationAgent
from agents.orchestrator import AgentOrchestrator
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from indexing.indexing_pipeline import IndexingPipeline
from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.document_loader import DocumentLoader
from ingestion.table_extractor import TableExtractor
from ingestion.table_summarizer import TableSummarizer
from ingestion.text_extractor import TextExtractor
from llm.llm_config import LLMConfig
from llm.openrouter_llm import OpenRouterLLM
from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from tools.calculator_tool import calculator
from utils.logger import logger
from vector_db.chroma_db import ChromaDB
from exceptions.custom_errors import ConfigurationError


def main():
    try:
        _run_repl(*_initialize_system())
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise SystemExit(1)


def _initialize_system():
    load_dotenv()

    env = {k: os.getenv(k) for k in (
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "EMBEDDING_MODEL", "RERANKER_MODEL", "MEMORY_DB_PATH",
    )}
    missing = [k for k, v in env.items() if not v]
    if missing:
        logger.error(f"Missing env vars: {missing}")
        raise SystemExit(1)

    try:
        config = LLMConfig(
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            top_p=float(os.getenv("LLM_TOP_P", 0.9)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 512)),
        )
        score_threshold = float(t) if (t := os.getenv("RERANKER_THRESHOLD")) else None
        retrieval_k = int(os.getenv("RETRIEVAL_K", 20))
        rerank_k = int(os.getenv("RERANK_K", 5))
    except (ValueError, ConfigurationError) as e:
        logger.error(f"Invalid config: {e}")
        raise SystemExit(1)

    llm = OpenRouterLLM(api_key=env["OPENROUTER_API_KEY"], model_name=env["OPENROUTER_MODEL"])
    embedding_model = SentenceTransformerEmbedding(model_name=env["EMBEDDING_MODEL"])
    vector_db = ChromaDB()

    pipeline = IndexingPipeline(
        loader=DocumentLoader(
            chunker=TextChunker(), text_extractor=TextExtractor(),
            table_extractor=TableExtractor(), chunk_factory=ChunkFactory(),
            table_summarizer=TableSummarizer(),
        ),
        embedding_model=embedding_model,
        vector_db=vector_db,
    )
    retriever = Retriever(
        embedding_model=embedding_model, vector_db=vector_db,
        reranker=Reranker(model_name=env["RERANKER_MODEL"]),
        retrieval_k=retrieval_k, rerank_k=rerank_k, score_threshold=score_threshold,
    )
    memory_manager = MemoryManager(
        memory=EpisodicMemory(db_path=env["MEMORY_DB_PATH"]),
        summarization_agent=SummarizationAgent(llm=llm),
    )
    orchestrator = AgentOrchestrator(
        retriever=retriever,
        memory_manager=memory_manager,
        reasoning_agent=ReasoningAgent(llm=llm),
        calculator_tool=calculator,
        llm=llm,
        llm_config=config,
    )

    runtime = RuntimeState()
    runtime.initialize(memory_manager=memory_manager)
    memory_manager.register_session(user_id=runtime.user_id, session_id=runtime.session_id)

    return pipeline, orchestrator, memory_manager, runtime


def _run_repl(pipeline, orchestrator, memory_manager, runtime):
    user_id, session_id = runtime.user_id, runtime.session_id
    interaction_count = 0

    file_path = input("Enter the path of the document to index: ").strip()
    if file_path:
        pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)

    print("\nSystem ready. Use /ingest <path> to add more documents.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue

        if query.lower() in ("exit", "bye"):
            if memory_manager.get_recent_interactions(user_id=user_id, session_id=session_id, limit=1):
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=True)
            print("Goodbye!")
            break

        if query.startswith("/ingest"):
            path = query[7:].strip()
            if path:
                try:
                    pipeline.index_document(file_path=path, user_id=user_id, session_id=session_id)
                    print(f"[System]: '{path}' indexed successfully.")
                except Exception as e:
                    print(f"[Error]: {e}")
            else:
                print("[System]: Usage: /ingest <file_path>")
            continue

        if query.lower() == "summary":
            summaries = memory_manager.get_summaries(user_id=user_id, session_id=session_id)
            print(f"\nSummary:\n{summaries[0][0]}" if summaries else "\nNo summaries yet.")
            continue

        try:
            answer = orchestrator.get_response(
                query=query, user_id=user_id, session_id=session_id
            )
        except Exception as e:
            print(f"\n[Error]: {e}")
            continue

        if not answer.strip():
            print("Assistant: I'm sorry, I encountered an issue. Please try again.")
            continue

        print(f"\nAssistant: {answer}\n")

        if memory_manager.should_store(query, answer):
            memory_manager.save_interaction(
                user_id=user_id, session_id=session_id, query=query, answer=answer
            )

        interaction_count += 1
        if interaction_count % 3 == 0:
            memory_manager.summarize_in_background(
                user_id=user_id, session_id=session_id, blocking=False
            )


if __name__ == "__main__":
    main()