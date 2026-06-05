import os
from dotenv import load_dotenv

from config.runtime_state import RuntimeState
from utils.logger import logger

from agents.reasoning_agent import ReasoningAgent
from agents.summarization_agent import SummarizationAgent
from agents.tool_agent import ToolAgent
from agents.query_router import QueryRouter
from agents.orchestrator import Orchestrator

from llm.openrouter_llm import OpenRouterLLM

from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from vector_db.chroma_db import ChromaDB
from indexing.indexing_pipeline import IndexingPipeline

from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.document_loader import DocumentLoader
from ingestion.table_extractor import TableExtractor
from ingestion.table_summarizer import TableSummarizer
from ingestion.text_extractor import TextExtractor

from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager

load_dotenv()

# -------------------------
# INGESTION PIPELINE
# -------------------------
chunker = TextChunker()
text_extractor = TextExtractor()
table_extractor = TableExtractor()
chunk_factory = ChunkFactory()
table_summarizer = TableSummarizer()

loader = DocumentLoader(
    chunker=chunker,
    text_extractor=text_extractor,
    table_extractor=table_extractor,
    chunk_factory=chunk_factory,
    table_summarizer=table_summarizer,
)

embedding_model = SentenceTransformerEmbedding()
vector_db = ChromaDB()

pipeline = IndexingPipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_db=vector_db
)

# -------------------------
# RETRIEVER
# -------------------------
reranker = Reranker(model_name=os.getenv("RERANKER_MODEL"))

retriever = Retriever(
    embedding_model=embedding_model,
    vector_db=vector_db,
    reranker=reranker,
    retrieval_k=20,
    rerank_k=5
)

# -------------------------
# LLM
# -------------------------
llm = OpenRouterLLM(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name=os.getenv("OPENROUTER_MODEL")
)

# -------------------------
# AGENTS
# -------------------------
tool_agent = ToolAgent(llm=llm)
reasoning_agent = ReasoningAgent(llm=llm)
summarization_agent = SummarizationAgent(llm=llm)
query_router = QueryRouter(llm=llm)

# -------------------------
# MEMORY
# -------------------------
memory = EpisodicMemory(db_path=os.getenv("MEMORY_DB_PATH"))

memory_manager = MemoryManager(
    memory=memory,
    summarization_agent=summarization_agent
)

runtime = RuntimeState()
runtime.initialize(memory_manager=memory_manager)

user_id = runtime.user_id
session_id = runtime.session_id

memory_manager.register_session(user_id=user_id, session_id=session_id)

# -------------------------
# ORCHESTRATOR
# -------------------------
orchestrator = Orchestrator(
    router=query_router,
    retriever=retriever,
    tool_agent=tool_agent,
    reasoning_agent=reasoning_agent,
    memory_manager=memory_manager
)

# -------------------------
# INDEX DOCUMENT
# -------------------------
file_path = input("Enter the path of the document to index: ")

logger.info(f"Started indexing: {file_path}")
pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)
logger.info(f"Indexing completed: {file_path}")

# -------------------------
# CHAT LOOP
# -------------------------
interaction_count = 0

while True:
    query = input("\nYou: ").strip()

    logger.info(f"\nUSER QUERY: {query}")

    # -------------------------
    # EXIT FLOW
    # -------------------------
    if query.lower() in ["exit", "bye"]:
        context = memory_manager.build_context(
            user_id=user_id,
            session_id=session_id,
            limit=100
        )

        summary = summarization_agent.summarize(context)

        memory_manager.save_summary(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            facts=""
        )

        logger.info("Session summary saved on exit")
        print("\nGoodbye!")
        break

    # -------------------------
    # MANUAL SUMMARY COMMAND
    # -------------------------
    if query.lower() == "summary":
        summaries = memory_manager.get_summaries(
            user_id=user_id,
            session_id=session_id
        )

        if summaries:
            print("\nSummary:\n")
            print("\n\n".join(summary for summary, _ in reversed(summaries)))
        else:
            print("\nNo summary available.")

        continue

    # -------------------------
    # MAIN ORCHESTRATION CALL
    # -------------------------
    answer = orchestrator.run(query, user_id, session_id)

    print(f"\nAssistant: {answer}")

    logger.info(f"FINAL ANSWER: {answer}")

    # -------------------------
    # MEMORY STORAGE CONTROL
    # -------------------------
    interaction_count += 1

    if interaction_count % 3 == 0:
        memory_manager.summarize_in_background(
            user_id=user_id,
            session_id=session_id
        )