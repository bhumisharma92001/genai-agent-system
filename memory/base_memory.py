from abc import ABC, abstractmethod

class BaseMemory(ABC):

    @abstractmethod
    def create_or_update_session(self, user_id: str, session_id: str) -> None:
        pass

    @abstractmethod
    def save_interaction(
        self, user_id: str, session_id: str, query: str, answer: str, importance_score: int = 1
    ) -> None:
        pass

    @abstractmethod
    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = 5) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def get_interactions_after_timestamp(self, user_id: str, session_id: str, timestamp: str) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def get_relevant_interactions(self, user_id: str, session_id: str, query: str, limit: int = 5) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def get_summaries(self, user_id: str, session_id: str) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def save_summary(self, user_id: str, session_id: str, summary: str) -> None:
        pass

    @abstractmethod
    def get_last_session_id(self, user_id: str) -> str | None:
        pass