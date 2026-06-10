from memory.base_memory import BaseMemory
from memory.memory_repository import MemoryRepository

class EpisodicMemory(BaseMemory):
    def __init__(self, db_path: str):
        self.repository = MemoryRepository(db_path=db_path)
        self.repository.initialize()

    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        self.repository.create_or_update_session(user_id=user_id, session_id=session_id)

    def save_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        importance_score: int = 1
    ):
        self.repository.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query=query,
            answer=answer,
            importance_score=importance_score
        )

    def get_recent_interactions(
        self, user_id: str, session_id: str, limit: int = 5
    ) -> list[tuple[str, str]]:
        return self.repository.get_recent_interactions(
            user_id=user_id, session_id=session_id, limit=limit
        )

    def get_interactions_after_timestamp(
        self, user_id: str, session_id: str, timestamp: str
    ) -> list[tuple[str, str]]:
        return self.repository.get_interactions_after_timestamp(
            user_id=user_id, session_id=session_id, timestamp=timestamp
        )

    def get_relevant_interactions(
        self, user_id: str, session_id: str, query: str, limit: int = 5
    ) -> list[tuple[str, str]]:
        return self.repository.get_relevant_interactions(
            user_id=user_id, session_id=session_id, query=query, limit=limit
        )

    def get_summaries(self, user_id: str, session_id: str) -> list[tuple[str, str]]:
        return self.repository.get_summaries(user_id=user_id, session_id=session_id)

    def save_summary(self, user_id: str, session_id: str, summary: str) -> None:
        self.repository.save_summary(user_id=user_id, session_id=session_id, summary=summary)

    def get_last_session_id(self, user_id: str) -> str | None:
        return self.repository.get_last_session_id(user_id=user_id)