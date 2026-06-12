import chromadb
from vector_db.base_vector_db import BaseVectorDB
from utils.logger import logger
from exceptions.custom_errors import VectorDBError, InvalidInputError, ModelLoadError


class ChromaDB(BaseVectorDB):

    def __init__(self, collection_name: str = "documents", persist_path: str = "vector_store"):
        if not collection_name or not collection_name.strip():
            raise InvalidInputError("collection_name cannot be empty")
        if not persist_path or not persist_path.strip():
            raise InvalidInputError("persist_path cannot be empty")
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized: collection='{collection_name}', path='{persist_path}'")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise ModelLoadError(f"Failed to initialize ChromaDB: {e}") from e

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            raise InvalidInputError("ids cannot be empty")
        if not embeddings:
            raise InvalidInputError("embeddings cannot be empty")
        if not documents:
            raise InvalidInputError("documents cannot be empty")
        if not metadatas:
            raise InvalidInputError("metadatas cannot be empty")
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise InvalidInputError("All input lists must have the same length")
        try:
            self.collection.upsert(
                ids=ids, embeddings=embeddings,
                documents=documents, metadatas=metadatas
            )
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"ChromaDB upsert failed: {str(e)}")
            raise VectorDBError(f"Failed to upsert vectors: {e}") from e

    def query(self, embedding: list[float], k: int = 5, where: dict | None = None) -> dict:
        if not embedding:
            raise InvalidInputError("embedding cannot be empty")
        if k <= 0:
            raise InvalidInputError("k must be positive")
        try:
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            raise VectorDBError(f"Failed to query vectors: {e}") from e

    def delete(self, where: dict) -> None:
        if not where:
            raise InvalidInputError("where filter cannot be empty")
        try:
            self.collection.delete(where=where)
            logger.info(f"ChromaDB delete successful: {where}")
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {str(e)}")
            raise VectorDBError(f"Failed to delete vectors: {e}") from e

    def list_sources(self, where: dict | None = None) -> list[str]:
        try:
            results = self.collection.get(where=where, include=["metadatas"])
            metadatas = results.get("metadatas") or []
            sources = list({m.get("source", "Unknown") for m in metadatas if m})
            return sorted(sources)
        except Exception as e:
            logger.error(f"ChromaDB list_sources failed: {str(e)}")
            raise VectorDBError(f"Failed to list sources: {e}") from e