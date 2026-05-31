from abc import ABC, abstractmethod


class BaseMemory(ABC):

    @abstractmethod
    def save_interaction(
        self,
        query: str,
        answer: str
    ) -> None:
        pass

    @abstractmethod
    def get_recent_interactions(
        self,
        limit: int = 5
    )-> list:
        pass

    @abstractmethod
    def get_summaries(self):
        pass

    @abstractmethod
    def save_summary(
        self,
        summary: str,
        facts: str
    ) -> None:
        pass