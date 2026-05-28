import chromadb

from vector_db.base_vector_db import (
    BaseVectorDB
)


class ChromaDB(BaseVectorDB):

    def __init__(
        self,
        collection_name: str = "documents"
    ):

        self.client = chromadb.PersistentClient(
            path="vector_store"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def upsert(
        self,
        ids,
        embeddings,
        documents,
        metadatas
    ):

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(
        self,
        embedding,
        k: int = 5
    ):

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )