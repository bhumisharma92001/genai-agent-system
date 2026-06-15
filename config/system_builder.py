import os
from agents.reasoning_agent import ReasoningAgent
from agents.summarization_agent import SummarizationAgent
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from indexing.indexing_pipeline import IndexingPipeline
from ingestion.chunk_factory import ChunkFactory
from ingestion.document_loader import DocumentLoader
from ingestion.table_extractor import TableExtractor
from ingestion.text_extractor import TextExtractor
from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from tools.calculator_tool import calculator
from tools.tool_registry import ToolRegistry
from orchestrator import AgentOrchestrator
from vector_db.chroma_db import ChromaDB
from agents.react_agent import ReActAgent
from utils.logger import logger
from functools import partial
from llm.openrouter_llm import build_llm
from agents.architect_agent import ArchitectAgent
from agents.chunking_agent import ChunkingAgent

def build_loader(llm, embedding_model) -> DocumentLoader:
    architect_agent = ArchitectAgent(llm=llm)
    chunking_agent = ChunkingAgent(
        embedding_model=embedding_model,
        architect_agent=architect_agent
    )
    return DocumentLoader(
        text_extractor=TextExtractor(),
        table_extractor=TableExtractor(),
        chunk_factory=ChunkFactory(),
        chunking_agent=chunking_agent,
    )


def build_retriever(embedding_model, vector_db) -> Retriever:
    reranker_model = os.getenv("RERANKER_MODEL")
    if not reranker_model:
        logger.error("RERANKER_MODEL not set in .env")
        raise SystemExit(1)

    env_threshold = os.getenv("RERANKER_THRESHOLD")
    try:
        score_threshold = float(env_threshold) if env_threshold is not None else None
    except ValueError:
        logger.error(f"Invalid RERANKER_THRESHOLD value: '{env_threshold}'.")
        raise SystemExit(1)

    try:
        retrieval_k = int(os.getenv("RETRIEVAL_K", 20))
        rerank_k = int(os.getenv("RERANK_K", 5))
    except ValueError:
        logger.error("RETRIEVAL_K and RERANK_K must be integers.")
        raise SystemExit(1)

    return Retriever(
        embedding_model=embedding_model,
        vector_db=vector_db,
        reranker=Reranker(model_name=reranker_model),
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        score_threshold=score_threshold
    )


def build_system() -> tuple:

    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    if not embedding_model_name:
        logger.error("EMBEDDING_MODEL not set in .env")
        raise SystemExit(1)

    memory_db_path = os.getenv("MEMORY_DB_PATH")
    if not memory_db_path:
        logger.error("MEMORY_DB_PATH not set in .env")
        raise SystemExit(1)

    embedding_model = SentenceTransformerEmbedding(model_name=embedding_model_name)
    vector_db = ChromaDB()
    llm = build_llm()
    pipeline = IndexingPipeline(
        loader=build_loader(llm=llm, embedding_model=embedding_model),
        embedding_model=embedding_model,
        vector_db=vector_db,
    )
    retriever = build_retriever(embedding_model, vector_db)

    memory_vector_db = ChromaDB(
        collection_name="memory_interactions",
        persist_path="memory_store"
    )
    memory_manager = MemoryManager(
        memory=EpisodicMemory(
            db_path=memory_db_path,
            embedding_model=embedding_model,
            vector_db=memory_vector_db
        ),
        summarization_agent=SummarizationAgent(llm=llm)
    )

    registry = ToolRegistry()
    registry.register("calculator", partial(calculator, llm=llm))
    react_agent = ReActAgent(llm=llm,retriever=retriever, registry=registry)

    orchestrator = AgentOrchestrator(
        retriever=retriever,
        memory_manager=memory_manager,
        reasoning_agent=ReasoningAgent(llm=llm),
        react_agent=react_agent,
        registry=registry,
        llm=llm,
    )

    return pipeline, orchestrator, memory_manager