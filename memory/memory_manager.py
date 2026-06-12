from utils.logger import logger
from memory.base_memory import BaseMemory
from memory.interaction_filter import InteractionFilter
from memory.memory_summarizer import MemorySummarizer
from agents.summarization_agent import SummarizationAgent


class MemoryManager:

    def __init__(self, memory: BaseMemory, summarization_agent: SummarizationAgent):
        self.memory = memory
        self._filter = InteractionFilter()
        self._summarizer = MemorySummarizer(
            summarization_agent=summarization_agent,
            memory=memory
        )

    def register_session(self, user_id: str, session_id: str) -> None:
        self.memory.create_or_update_session(user_id=user_id, session_id=session_id)

    def summarize_in_background(self, user_id: str, session_id: str, blocking: bool = False) -> None:
        self._summarizer.run(user_id=user_id, session_id=session_id, blocking=blocking)

    def save_interaction(self, user_id: str, session_id: str, query: str, answer: str) -> None:
        importance = self._filter.get_importance(query, answer)
        self.memory.save_interaction(
            user_id=user_id, session_id=session_id,
            query=query, answer=answer, importance_score=importance
        )

    def should_store(self, query: str, answer: str) -> bool:
        return self._filter.should_store(query, answer)

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = 5) -> list:
        return self.memory.get_recent_interactions(
            user_id=user_id, session_id=session_id, limit=limit
        )

    def get_summaries(self, user_id: str, session_id: str) -> list:
        return self.memory.get_summaries(user_id=user_id, session_id=session_id)

    def get_last_session_id(self, user_id: str) -> str | None:
        return self.memory.get_last_session_id(user_id=user_id)

    def get_all_sessions(self, user_id: str) -> list[tuple[str, str, str]]:
        return self.memory.get_all_sessions(user_id=user_id)

    def get_relevant_memory(self, query: str, user_id: str, session_id: str) -> list:
        interactions = self.memory.get_relevant_interactions(
            user_id=user_id, session_id=session_id, query=query
        )
        summaries = self.memory.get_summaries(user_id=user_id, session_id=session_id)
        if summaries:
            summary_turn = ("Previous conversation summary", summaries[0][0])
            return [summary_turn] + interactions
        return interactions