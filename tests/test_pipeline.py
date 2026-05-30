import os

from dotenv import load_dotenv

from ingestion.chunker import TextChunker
from ingestion.document_loader import DocumentLoader

from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from vector_db.chroma_db import ChromaDB

from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

from indexing.indexing_pipeline import (
    IndexingPipeline
)

from agents.reasoning_agent import (
    ReasoningAgent
)

from llm.gemini_llm import GeminiLLM

from llm.llm_config import LLMConfig


# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY not found"
    )


# --------------------------------------------------
# SETUP
# --------------------------------------------------

print("\n" + "=" * 100)
print("INITIALIZING COMPONENTS")
print("=" * 100)

chunker = TextChunker(
    chunk_size=700,
    overlap=150
)

loader = DocumentLoader(
    chunker=chunker
)

embedding_model = (
    SentenceTransformerEmbedding()
)

vector_db = ChromaDB()

reranker = Reranker()

llm = GeminiLLM(
    api_key=api_key
)

config = LLMConfig()

pipeline = IndexingPipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_db=vector_db
)

retriever = Retriever(
    embedding_model=embedding_model,
    vector_db=vector_db,
    reranker=reranker
)

reasoning_agent = (
    ReasoningAgent(
        llm=llm
    )
)


# --------------------------------------------------
# LOAD DOCUMENT
# --------------------------------------------------

print("\n" + "=" * 100)
print("DOCUMENT LOADING")
print("=" * 100)

document_path = (
    r"C:\Users\bhoomi.sharma\Downloads\advanced_ml_enterprise_report.pdf"
)

chunks = loader.load(
    document_path
)

print(
    f"\nTOTAL CHUNKS CREATED: {len(chunks)}"
)

first_chunk = chunks[0]

print("\nFIRST CHUNK ID:\n")
print(
    first_chunk["chunk_id"]
)

print("\nFIRST CHUNK TYPE:\n")
print(
    first_chunk["chunk_type"]
)

print("\nFIRST CHUNK TEXT:\n")
print(
    first_chunk["text"][:500]
)


# --------------------------------------------------
# EMBEDDING TEST
# --------------------------------------------------

print("\n" + "=" * 100)
print("EMBEDDING TEST")
print("=" * 100)

embedding = (
    embedding_model.embed(
        first_chunk["text"]
    )
)

print("\nEMBEDDING DIMENSION:\n")
print(
    len(embedding)
)

print("\nFIRST 10 VALUES:\n")
print(
    embedding[:10]
)


# --------------------------------------------------
# INDEXING
# --------------------------------------------------

print("\n" + "=" * 100)
print("INDEXING DOCUMENT")
print("=" * 100)

pipeline.index_document(
    document_path
)

print(
    "\nDOCUMENT INDEXED SUCCESSFULLY"
)


# --------------------------------------------------
# CHROMA CHECK
# --------------------------------------------------

print("\n" + "=" * 100)
print("CHROMA STORAGE CHECK")
print("=" * 100)

total_vectors = (
    vector_db.collection.count()
)

print(
    f"\nTOTAL VECTORS: {total_vectors}"
)

stored = (
    vector_db.collection.get()
)

print("\nFIRST STORED DOCUMENT:\n")
print(
    stored["documents"][0][:500]
)

print("\nFIRST STORED METADATA:\n")
print(
    stored["metadatas"][0]
)


# --------------------------------------------------
# QUERY TESTS
# --------------------------------------------------

queries = [
    "What is hybrid retrieval?",
    "What is semantic search?",
    "Who won FIFA World Cup 2022?",
    "What is the capital of France?"
]

for query in queries:

    print("\n")
    print("=" * 100)
    print("NEW QUERY")
    print("=" * 100)

    print("\nQUERY:\n")
    print(query)

    query_embedding = (
        embedding_model.embed(query)
    )

    print("\nQUERY VECTOR DIMENSION:\n")
    print(
        len(query_embedding)
    )

    print("\nQUERY VECTOR SAMPLE:\n")
    print(
        query_embedding[:10]
    )

    raw_results = (
        vector_db.query(
            embedding=query_embedding,
            k=5
        )
    )

    print("\nCHROMA DISTANCES:\n")
    print(
        raw_results["distances"][0]
    )

    chunks = retriever.retrieve(
        query=query
    )

    print("\nRETRIEVED CHUNKS:\n")

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nCHUNK {index}"
        )

        print("\nSCORE:\n")
        print(
            chunk["score"]
        )

        print("\nTEXT:\n")
        print(
            chunk["text"][:500]
        )

    context = "\n\n".join(
        chunk["text"]
        for chunk in chunks
    )

    print("\nCONTEXT SENT TO LLM:\n")
    print(
        context[:2000]
    )

    answer = reasoning_agent.answer(
        query=query,
        chunks=chunks,
        config=config
    )

    print("\nFINAL ANSWER:\n")
    print(answer)

    print("\n" + "=" * 100)


print("\n")
print("=" * 100)
print("PIPELINE TEST COMPLETED")
print("=" * 100)