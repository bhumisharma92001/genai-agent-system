import hashlib
from memory.base_memory import BaseMemory
from memory.memory_repository import MemoryRepository
from embeddings.base import BaseEmbedding
from vector_db.base_vector_db import BaseVectorDB
from utils.logger import logger


class EpisodicMemory(BaseMemory):

    def __init__(self, db_path: str, embedding_model: BaseEmbedding, vector_db: BaseVectorDB):
        self.repository = MemoryRepository(db_path=db_path)
        self.repository.initialize()
        self._embedding_model = embedding_model
        self._vector_db = vector_db

    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        self.repository.create_or_update_session(user_id=user_id, session_id=session_id)

    def save_interaction(
        self, user_id: str, session_id: str, query: str, answer: str, importance_score: int = 1
    ) -> None:
        self.repository.save_interaction(
            user_id=user_id, session_id=session_id,
            query=query, answer=answer, importance_score=importance_score
        )
        try:
            self._vector_db.upsert(
                ids=[f"{user_id}:{session_id}:{hashlib.md5(query.encode()).hexdigest()}"],
                embeddings=[self._embedding_model.embed(query, is_query=False)],
                documents=[f"Q: {query}\nA: {answer}"],
                metadatas=[{"user_id": user_id, "session_id": session_id, "importance_score": importance_score}]
            )
        except Exception as e:
            logger.warning(f"Memory vector upsert failed (non-fatal): {e}")

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = 5) -> list[tuple[str, str]]:
        return self.repository.get_recent_interactions(
            user_id=user_id, session_id=session_id, limit=limit
        )

    def get_interactions_after_id(self, user_id: str, session_id: str, last_id: int) -> list[tuple[str, str]]:
        return self.repository.get_interactions_after_id(
            user_id=user_id, session_id=session_id, last_id=last_id
        )

    def get_relevant_interactions(self, user_id: str, session_id: str, query: str, limit: int = 5) -> list[tuple[str, str]]:
        try:
            results = self._vector_db.query(
                embedding=self._embedding_model.embed(query, is_query=True),
                k=limit,
                where={"$and": [{"user_id": {"$eq": user_id}}, {"session_id": {"$eq": session_id}}]}
            )
            docs = results.get("documents") or []
            docs = docs[0] if docs and isinstance(docs[0], list) else docs
            interactions = []
            for doc in docs:
                parts = doc.split("\nA: ", 1)
                if len(parts) == 2:
                    interactions.append((parts[0].replace("Q: ", "", 1), parts[1]))
            if interactions:
                return interactions
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to SQL: {e}")
        logger.info("Falling back to SQL-based retrieval...")
        return self.repository.get_relevant_interactions(
            user_id=user_id, session_id=session_id, query=query, limit=limit
        )

    def get_summaries(self, user_id: str, session_id: str) -> list[tuple[str, int]]:
        return self.repository.get_summaries(user_id=user_id, session_id=session_id)

    def save_summary(self, user_id: str, session_id: str, summary: str, last_summarized_id: int) -> None:
        self.repository.save_summary(
            user_id=user_id, session_id=session_id,
            summary=summary, last_summarized_id=last_summarized_id
        )

    def get_last_session_id(self, user_id: str) -> str | None:
        return self.repository.get_last_session_id(user_id=user_id)

    def get_all_sessions(self, user_id: str) -> list[tuple[str, str, str]]:
        return self.repository.get_all_sessions(user_id=user_id)

    def get_max_interaction_id(self, user_id: str, session_id: str) -> int:
        return self.repository.get_max_interaction_id(
            user_id=user_id, session_id=session_id
        )