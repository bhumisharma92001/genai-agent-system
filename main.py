import os

from dotenv import load_dotenv

from ingestion.chunker import TextChunker
from ingestion.document_loader import DocumentLoader

from indexing.indexing_pipeline import (
    IndexingPipeline
)

from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from llm.openrouter_llm import OpenRouterLLM
from vector_db.chroma_db import ChromaDB

from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

from agents.reasoning_agent import ReasoningAgent

from llm.llm_config import LLMConfig

from memory.episodic_memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from agents.summarization_agent import (
    SummarizationAgent
)

load_dotenv()


# -------------------------
# INGESTION
# -------------------------

chunker = TextChunker()

loader = DocumentLoader(
    chunker=chunker
)

embedding_model = (
    SentenceTransformerEmbedding()
)

vector_db = ChromaDB()

pipeline = IndexingPipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_db=vector_db
)

# -------------------------
# DOCUMENT INDEXING
# -------------------------

file_path = input(
    "Enter the path of the document to index:"
)

pipeline.index_document(
    file_path=file_path
)

print(
    "\nDocument indexed successfully."
)

# -------------------------
# RETRIEVAL
# -------------------------

reranker = Reranker()

retriever = Retriever(
    embedding_model=embedding_model,
    vector_db=vector_db,
    reranker=reranker
)

# -------------------------
# LLM
# -------------------------


llm = OpenRouterLLM(
    api_key=os.getenv("OPENROUTER_API_KEY")
)

reasoning_agent = (
    ReasoningAgent(
        llm=llm
    )
)
summarization_agent = (
    SummarizationAgent(
        llm=llm
    )
)
config = LLMConfig()

# -------------------------
# MEMORY
# -------------------------

memory = EpisodicMemory()

memory_manager = MemoryManager(
    memory=memory
)

# -------------------------
# CHAT LOOP
# -------------------------

while True:

    query = input(
        "\nYou: "
    )

    if query.lower() in ["exit", "bye"]:

        print(
            "\nGenerating conversation summary..."
        )

        context = (
            memory_manager.build_context(
                limit=100
                )
            )
        

        summary = (
            summarization_agent.summarize(
                context
                )
            )

        facts = (
            summarization_agent.extract_facts(
                context
                )
            )

        memory_manager.save_summary(
            summary=summary,
            facts=facts
        )

        print("\n========== SUMMARY ==========")
        print(summary)

        print("\n========== FACTS ==========")
        print(facts)

        print("\n========== INTERACTIONS ==========")

        for row in memory_manager.get_recent_interactions(
            limit=100
        ):
            print(row)

        print("\nGoodbye!")

        break

    chunks = retriever.retrieve(
        query=query
    )

    answer = reasoning_agent.answer(
        query=query,
        chunks=chunks,
        config=config
    )

    memory_manager.save_interaction(
        query=query,
        answer=answer
    )

    print(
        f"\nAssistant: {answer}"
    )