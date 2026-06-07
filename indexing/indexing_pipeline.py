import os
import logging
from ingestion.document_loader import DocumentLoader
from embeddings.base import BaseEmbedding
from vector_db.base_vector_db import BaseVectorDB

logger = logging.getLogger(__name__)

class IndexingPipeline:

    def __init__(self, loader: DocumentLoader, embedding_model: BaseEmbedding, vector_db: BaseVectorDB):
        self.loader = loader
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def index_document(self, file_path: str, user_id: str, session_id: str):
        doc_id = os.path.basename(file_path)
        chunks = self.loader.load(file_path, user_id, session_id, document_id=doc_id)

        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        if not valid_chunks:
            logger.info("No valid chunks to index.")
            return

        texts = [c["text"].strip() for c in valid_chunks]
        embeddings = self.embedding_model.embed_batch(texts)

        self.vector_db.upsert(
            ids=[str(c["chunk_id"]) for c in valid_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in valid_chunks]
        )
        logger.info(f"Successfully indexed {len(valid_chunks)} chunks to Vector DB.")