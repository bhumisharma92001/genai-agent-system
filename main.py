import os
import re
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
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model_name:
        logger.error("OPENROUTER_API_KEY or OPENROUTER_MODEL not set in .env")
        raise SystemExit(1)
    
    # Ingestion & Indexing setup
    loader = DocumentLoader(
        chunker=TextChunker(),
        text_extractor=TextExtractor(),
        table_extractor=TableExtractor(),
        chunk_factory=ChunkFactory(),
        table_summarizer=TableSummarizer(),
    )
    embedding_model = SentenceTransformerEmbedding()
    vector_db = ChromaDB()
    pipeline = IndexingPipeline(loader=loader, embedding_model=embedding_model, vector_db=vector_db)
    
    # Retrieval setup
    reranker = Reranker(model_name=os.getenv("RERANKER_MODEL"))
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_db=vector_db,
        reranker=reranker,
        retrieval_k=20,
        rerank_k=5
    )
    
    # LLM & Multi-Agent Core
    llm = OpenRouterLLM(api_key=api_key, model_name=model_name)
    tool_agent = ToolAgent(llm=llm)
    reasoning_agent = ReasoningAgent(llm=llm)
    summarization_agent = SummarizationAgent(llm=llm)
    config = LLMConfig()
    
    # Session Execution Context & Memory Storage
    memory = EpisodicMemory(db_path=os.getenv("MEMORY_DB_PATH"))
    memory_manager = MemoryManager(memory=memory, summarization_agent=summarization_agent)
    runtime = RuntimeState()
    runtime.initialize(memory_manager=memory_manager)
    
    memory_manager.register_session(user_id=runtime.user_id, session_id=runtime.session_id)
    
    return pipeline, retriever, tool_agent, reasoning_agent, memory_manager, runtime, config

def run_repl_loop(pipeline, retriever, tool_agent, reasoning_agent, memory_manager, runtime, config):
    """Encapsulates the CLI Read-Eval-Print-Loop (REPL)."""
    user_id = runtime.user_id
    session_id = runtime.session_id
    
    file_path = input("Enter the path of the document to index: ").strip()
    if file_path:
        logger.info(f"Started indexing: {file_path}")
        pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)
    
    interaction_count = 0
    print("\nSystem ready. Type your queries below.")
    
    while True:
        raw_query = input("\nYou: ")
        query = raw_query.strip()
        if not query:
            continue
            
        logger.info(f"User query received: {query}")
        clean_query = query.lower()

        if clean_query in ["exit", "bye"]:
            recent_records = memory_manager.get_recent_interactions(user_id=user_id, session_id=session_id, limit=1)
            if recent_records:
                logger.info("Compiling final session summary state synchronously on exit.")
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=True)
            print("\nGoodbye!")
            break

        if clean_query == "summary":
            summaries = memory_manager.get_summaries(user_id=user_id, session_id=session_id)
            if summaries:
                print(f"\nSummary:\n{summaries[0][0]}")
            else:
                print("\nNo session summaries have been processed yet.")
            continue

        route = tool_agent.route(query)
        if route == "calculator":
            result = tool_agent.execute(query)
            answer = str(result.get("result", "Could not compute the result."))
        else:
            chunks = retriever.retrieve(query=query, user_id=user_id, session_id=session_id)
            conversation_history = memory_manager.get_relevant_memory(query=query, user_id=user_id, session_id=session_id)
            answer = reasoning_agent.answer(
                query=query, chunks=chunks, conversation_history=conversation_history, config=config
            )
            
        print(f"\nAssistant: {answer}")
        logger.info("Answer generated successfully.")

        if memory_manager.should_store(query, answer):
            memory_manager.save_interaction(user_id=user_id, session_id=session_id, query=query, answer=answer)
            logger.info("Interaction successfully committed to episodic storage.")
            
            interaction_count += 1
            if interaction_count % 3 == 0:
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)
        else:
            logger.info("Skipping tracking matrix for utility/low-importance execution loops.")
if __name__ == "__main__":
    main()