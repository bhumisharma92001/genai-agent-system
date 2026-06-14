from embeddings.base import BaseEmbedding
from retrieval.reranker import Reranker
from utils.logger import logger
from vector_db.base_vector_db import BaseVectorDB
from exceptions.custom_errors import RetrievalError, InvalidInputError


class Retriever:

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        reranker: Reranker,
        retrieval_k: int = 20,
        rerank_k: int = 5,
        score_threshold: float | None = None
    ):
        if rerank_k > retrieval_k:
            raise InvalidInputError("rerank_k cannot be greater than retrieval_k")
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, user_id: str, session_id: str) -> list[dict]:
        if not query or not query.strip():
            raise InvalidInputError("query cannot be empty")

        try:
            query_embedding = self.embedding_model.embed(query, is_query=True)
            results = self.vector_db.query(
                embedding=query_embedding,
                k=self.retrieval_k,
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"session_id": {"$eq": session_id}}
                    ]
                }
            )

            if not results:
                logger.warning(f"Vector DB returned empty payload for session: {session_id}")
                return []

            raw_documents = results.get("documents") or []
            raw_metadatas = results.get("metadatas") or []
            raw_distances = results.get("distances") or []

            documents = raw_documents[0] if raw_documents and isinstance(raw_documents[0], list) else raw_documents
            metadatas = raw_metadatas[0] if raw_metadatas and isinstance(raw_metadatas[0], list) else raw_metadatas
            distances = raw_distances[0] if raw_distances and isinstance(raw_distances[0], list) else raw_distances

            if not documents:
                logger.info("No documents found in Vector DB for the given session.")
                return []

            if len(documents) != len(metadatas):
                logger.warning(f"Metadata count mismatch: docs={len(documents)}, meta={len(metadatas)}. Padding with empty.")
                metadatas = [metadatas[i] if i < len(metadatas) else {} for i in range(len(documents))]

            chunks = []
            for i, (document, metadata) in enumerate(zip(documents, metadatas)):
                vector_distance = float(distances[i]) if i < len(distances) and distances[i] is not None else 1.0
                chunks.append({
                    "text": document,
                    "metadata": metadata,
                    "vector_distance": vector_distance
                })

            return self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=self.rerank_k,
                score_threshold=self.score_threshold
            )

        except InvalidInputError:
            raise
        except Exception as e:
            logger.exception("Failed to retrieve chunks")
            raise RetrievalError(f"Failed to retrieve chunks: {e}") from e