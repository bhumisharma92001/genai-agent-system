import os
from ingestion.document_loader import DocumentLoader
from embeddings.base import BaseEmbedding
from vector_db.base_vector_db import BaseVectorDB
from utils.logger import logger
from exceptions.custom_errors import IndexingError, InvalidInputError


class IndexingPipeline:

    def __init__(self, loader: DocumentLoader, embedding_model: BaseEmbedding, vector_db: BaseVectorDB):
        self.loader = loader
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def index_document(self, file_path: str, user_id: str, session_id: str) -> None:
        if not file_path or not file_path.strip():
            raise InvalidInputError("file_path cannot be empty")
        try:
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

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Indexing failed for file: {file_path} — {str(e)}")
            raise IndexingError(f"Failed to index document: {file_path}") from e

    def delete_document(self, source: str, user_id: str, session_id: str) -> None:
        if not source or not source.strip():
            raise InvalidInputError("source cannot be empty")
        try:
            existing = self.vector_db.list_sources(where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"session_id": {"$eq": session_id}}
                ]
            })
            if source not in existing:
                raise InvalidInputError(f"Document '{source}' not found in index.")
            self.vector_db.delete(where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"session_id": {"$eq": session_id}},
                    {"source": {"$eq": source}}
                ]
            })
            logger.info(f"Deleted document: '{source}'")
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Delete failed: {source} — {str(e)}")
            raise IndexingError(f"Failed to delete document: {source}") from e

    def list_documents(self, user_id: str, session_id: str) -> list[str]:
        try:
            return self.vector_db.list_sources(where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"session_id": {"$eq": session_id}}
                ]
            })
        except Exception as e:
            logger.error(f"List failed — {str(e)}")
            raise IndexingError(f"Failed to list documents: {e}") from e