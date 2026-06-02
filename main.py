import os

from dotenv import load_dotenv

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
from llm.openrouter_llm import OpenRouterLLM
from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from utils.logger import logger
from vector_db.chroma_db import ChromaDB
load_dotenv()

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

embedding_model = (SentenceTransformerEmbedding())
vector_db = ChromaDB()
pipeline = IndexingPipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_db=vector_db
)

file_path = input("Enter the path of the document to index:")
logger.info(f"Started indexing: {file_path}")
pipeline.index_document(file_path=file_path)
logger.info(f"Indexed document: {file_path}")
reranker = Reranker(model_name=os.getenv("RERANKER_MODEL"))
retriever = Retriever(
    embedding_model=embedding_model,
    vector_db=vector_db,
    reranker=reranker,
    retrieval_k=20,
    rerank_k=5
)
llm = OpenRouterLLM(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name=os.getenv("OPENROUTER_MODEL")
)
reasoning_agent = (ReasoningAgent(llm=llm))
summarization_agent = (SummarizationAgent(llm=llm))
config = LLMConfig()

memory = EpisodicMemory(db_path=os.getenv("MEMORY_DB_PATH"))
memory_manager = MemoryManager(memory=memory)

while True:
    query = input("\nYou: ")
    logger.info(f"User query: {query}")
    if query.lower() in ["exit", "bye"]:
        context = (memory_manager.build_context(limit=100))
        summary = (summarization_agent.summarize(context))
        facts = (summarization_agent.extract_facts(context))
        logger.info("Saving conversation summary")
        memory_manager.save_summary(summary=summary,facts=facts)              
        logger.info("Conversation summary saved")
        for row in memory_manager.get_recent_interactions(limit=100):
            print(row)
        print("\nGoodbye!")
        break

    chunks = retriever.retrieve(query=query)
    conversation_history = (memory_manager.get_recent_interactions(limit=10))
    answer = reasoning_agent.answer(
        query=query,
        chunks=chunks,
        conversation_history=conversation_history,
        config=config
    )
    print(f"\nAssistant: {answer}")
    logger.info(f"Answer generated for query: {query}")
    logger.info("Saving interaction")
    memory_manager.save_interaction(query=query,answer=answer)
    logger.info("Interaction saved")
