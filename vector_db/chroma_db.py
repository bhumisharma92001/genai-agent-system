import chromadb

from vector_db.base_vector_db import BaseVectorDB

class ChromaDB(BaseVectorDB):

    def __init__(self,collection_name: str = "documents",persist_path: str = "vector_store"):
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        if not persist_path.strip():
            raise ValueError("persist_path cannot be empty")

        try:
            self.client = chromadb.PersistentClient(path=persist_path)

            self.collection = (
                self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            )

        except Exception as e:
            raise RuntimeError("Failed to initialize ChromaDB") from e

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict]
    ) -> None:

        if not ids:
            raise ValueError("ids cannot be empty")

        if not embeddings:
            raise ValueError("embeddings cannot be empty")

        if not documents:
            raise ValueError("documents cannot be empty")

        if not metadatas:
            raise ValueError("metadatas cannot be empty")

        total_records = len(ids)

        if (
            len(embeddings) != total_records
            or len(documents) != total_records
            or len(metadatas) != total_records
        ):

            raise ValueError("All input lists must have the same length")

        try:

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

        except Exception as e:
            raise RuntimeError("Failed to upsert vectors") from e

    def query(self, embedding: list[float], k: int = 5, where: dict = None):
        if not embedding:
            raise ValueError("embedding cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        try:
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

        except Exception as e:
            raise RuntimeError("Failed to query vectors") from e