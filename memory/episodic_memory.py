from memory.base_memory import BaseMemory
from memory.memory_repository import MemoryRepository

class EpisodicMemory(BaseMemory):

    def __init__(self,db_path: str):
        self.repository = MemoryRepository(db_path=db_path)
        self.repository.initialize()

    def save_interaction(self,query: str,answer: str) -> None:
        self.repository.save_interaction(query=query,answer=answer)

    def get_recent_interactions(self,limit: int = 5) -> list[tuple[str, str]]:
        return self.repository.get_recent_interactions(limit=limit)

    def get_summaries(self) -> list[tuple[str, str]]:
        return self.repository.get_summaries()

    def save_summary(self,summary: str,facts: str) -> None:
        self.repository.save_summary(summary=summary,facts=facts)