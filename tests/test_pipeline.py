from ingestion.chunker import TextChunker

from ingestion.document_loader import (
    DocumentLoader
)

from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from vector_db.chroma_db import ChromaDB

from retrieval.reranker import Reranker

from retrieval.retriever import Retriever

from indexing.indexing_pipeline import (
    IndexingPipeline
)


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

# -----------------------------------
# INDEX DOCUMENT
# -----------------------------------

pipeline.index_document(
    r"C:\Users\bhoomi.sharma\Downloads\advanced_ml_enterprise_report.pdf"
)

# -----------------------------------
# QUERY
# -----------------------------------

results = retriever.retrieve(
    query="What is hybrid retrieval?"
)

for result in results:

    print("\n====================")
    print(result["text"])
    print(result["metadata"])
    print(result["score"])