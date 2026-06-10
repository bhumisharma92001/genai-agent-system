import os
from dotenv import load_dotenv

from config.runtime_state import RuntimeState
from agents.reasoning_agent import ReasoningAgent
from agents.summarization_agent import SummarizationAgent
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from indexing.indexing_pipeline import IndexingPipeline
from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.document_loader import DocumentLoader
from ingestion.table_extractor import TableExtractor
from ingestion.table_summarizer import TableSummarizer
from ingestion.text_extractor import TextExtractor
from llm.llm_config import LLMConfig
from agents.tool_agent import ToolAgent
from llm.openrouter_llm import OpenRouterLLM
from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from utils.logger import logger
from vector_db.chroma_db import ChromaDB
from exceptions.custom_errors import ConfigurationError

def main():
    try:
        components = initialize_system()
        run_repl_loop(*components)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise SystemExit(1)

def initialize_system():
    load_dotenv()

    # --- LLM ---
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model_name:
        logger.error("OPENROUTER_API_KEY or OPENROUTER_MODEL not set in .env")
        raise SystemExit(1)

    # --- Ingestion ---
    loader = DocumentLoader(
        chunker=TextChunker(),
        text_extractor=TextExtractor(),
        table_extractor=TableExtractor(),
        chunk_factory=ChunkFactory(),
        table_summarizer=TableSummarizer(),
    )

    # --- Embedding ---
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    if not embedding_model_name:
        logger.error("EMBEDDING_MODEL not set in .env")
        raise SystemExit(1)
    embedding_model = SentenceTransformerEmbedding(model_name=embedding_model_name)

    # --- Vector DB & Pipeline ---
    vector_db = ChromaDB()
    pipeline = IndexingPipeline(loader=loader, embedding_model=embedding_model, vector_db=vector_db)

    # --- Reranker & Retriever ---
    reranker_model = os.getenv("RERANKER_MODEL")
    if not reranker_model:
        logger.error("RERANKER_MODEL not set in .env")
        raise SystemExit(1)

    reranker = Reranker(model_name=reranker_model)

    env_threshold = os.getenv("RERANKER_THRESHOLD")
    try:
        score_threshold = float(env_threshold) if env_threshold is not None else None
    except ValueError:
        logger.error(f"Invalid RERANKER_THRESHOLD value: '{env_threshold}'. Must be a float.")
        raise SystemExit(1)

    try:
        retrieval_k = int(os.getenv("RETRIEVAL_K", 20))
        rerank_k = int(os.getenv("RERANK_K", 5))
    except ValueError:
        logger.error("RETRIEVAL_K and RERANK_K must be integers.")
        raise SystemExit(1)

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_db=vector_db,
        reranker=reranker,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        score_threshold=score_threshold
    )

    # --- LLM & Agents ---
    llm = OpenRouterLLM(api_key=api_key, model_name=model_name)
    tool_agent = ToolAgent(llm=llm)
    reasoning_agent = ReasoningAgent(llm=llm)
    summarization_agent = SummarizationAgent(llm=llm)

    try:
        config = LLMConfig(
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            top_p=float(os.getenv("LLM_TOP_P", 0.9)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 512))
        )
    except ConfigurationError as e:
        logger.error(f"Invalid LLM config: {e}")
        raise SystemExit(1)

    # --- Memory ---
    memory_db_path = os.getenv("MEMORY_DB_PATH")
    if not memory_db_path:
        logger.error("MEMORY_DB_PATH not set in .env")
        raise SystemExit(1)
    memory = EpisodicMemory(db_path=memory_db_path)
    memory_manager = MemoryManager(memory=memory, summarization_agent=summarization_agent)

    # --- Runtime ---
    runtime = RuntimeState()
    runtime.initialize(memory_manager=memory_manager)
    memory_manager.register_session(user_id=runtime.user_id, session_id=runtime.session_id)

    return pipeline, retriever, tool_agent, reasoning_agent, memory_manager, runtime, config


def run_repl_loop(pipeline, retriever, tool_agent, reasoning_agent, memory_manager, runtime, config):
    user_id = runtime.user_id
    session_id = runtime.session_id

    file_path = input("Enter the path of the document to index: ").strip()
    if file_path:
        logger.info(f"Started indexing: {file_path}")
        pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)

    interaction_count = 0
    print("\nSystem ready. Type your queries below.")
    print("To index a new document at any time, use: /ingest <file_path>")

    while True:
        raw_query = input("\nYou: ")
        query = raw_query.strip()
        if not query:
            continue

        logger.info(f"User query received: {query}")
        clean_query = query.lower()
        answer = ""

        # --- 1. Exit ---
        if clean_query in ["exit", "bye"]:
            recent_records = memory_manager.get_recent_interactions(user_id=user_id, session_id=session_id, limit=1)
            if recent_records:
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=True)
            print("\nGoodbye!")
            break

        # --- 2. Ingest ---
        if query.startswith("/ingest"):
            _handle_ingest(query, pipeline, user_id, session_id, memory_manager, interaction_count)
            interaction_count += 1
            continue

        # --- 3. Summary ---
        if clean_query == "summary":
            summaries = memory_manager.get_summaries(user_id=user_id, session_id=session_id)
            print(f"\nSummary:\n{summaries[0][0]}" if summaries else "\nNo session summaries yet.")
            continue

        # --- 4. Route & Execute ---
        route = tool_agent.route(query)
        if "calculator" in route:
            result = tool_agent.execute(query)
            answer = str(result.get("result", "Could not compute the result."))
        else:
            chunks = retriever.retrieve(query=query, user_id=user_id, session_id=session_id)
            conversation_history = memory_manager.get_relevant_memory(query=query, user_id=user_id, session_id=session_id)
            answer = reasoning_agent.answer(
                query=query, chunks=chunks, history=conversation_history, config=config
            ) or ""

        if not answer.strip():
            print("\nAssistant: I'm sorry, I encountered an issue. Please try again.")
            logger.warning("Empty answer returned. Skipping storage.")
            continue

        print(f"\nAssistant: {answer}")
        logger.info("Answer generated successfully.")

        # --- 5. Store ---
        if memory_manager.should_store(query, answer):
            memory_manager.save_interaction(user_id=user_id, session_id=session_id, query=query, answer=answer)
            logger.info("Interaction committed to episodic storage.")
        else:
            logger.info("Skipping low-importance interaction.")

        interaction_count += 1
        if interaction_count % 3 == 0:
            memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)


def _handle_ingest(query, pipeline, user_id, session_id, memory_manager, interaction_count):
    target_path = query[7:].strip()
    if target_path:
        try:
            pipeline.index_document(file_path=target_path, user_id=user_id, session_id=session_id)
            print(f"\n[System]: Document '{target_path}' successfully indexed.")
        except Exception as e:
            print(f"\n[Error]: Failed to index document: {e}")
    else:
        print("\n[System]: Please provide a valid file path. Usage: /ingest <path>")

    if (interaction_count + 1) % 3 == 0:
        memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)


if __name__ == "__main__":
    main()