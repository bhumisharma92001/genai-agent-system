from memory.base_memory import BaseMemory
import threading
from utils.logger import logger

class MemoryManager:

    def __init__(self, memory: BaseMemory, summarization_agent):
        self.memory = memory
        self.summarization_agent = summarization_agent

    def register_session(self, user_id: str, session_id: str) -> None:
        self.memory.create_or_update_session(user_id=user_id, session_id=session_id)

    def summarize_in_background(self, user_id: str, session_id: str):
        thread = threading.Thread(
            target=self._generate_summary,
            args=(user_id, session_id),
            daemon=True
        )
        thread.start()

    def _generate_summary(self, user_id: str, session_id: str):
        try:
            interactions = self.memory.get_recent_interactions(user_id=user_id, session_id=session_id, limit=3)
            if not interactions:
                return

            conversation = "\n".join(f"User: {query}\nAssistant: {answer}" for query, answer in interactions)
            summary = self.summarization_agent.summarize(conversation)
            self.memory.save_summary(
                user_id=user_id,
                session_id=session_id,
                summary=summary,
                facts=""
            )

        except Exception:
            logger.exception("Background summarizer failed")

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = 5) -> list:
        return self.memory.get_recent_interactions(
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )

    def get_last_session_id(self, user_id: str) -> str | None:
        return self.memory.get_last_session_id(user_id=user_id)

    def get_summaries(self, user_id: str, session_id: str) -> list:
        return self.memory.get_summaries(
            user_id=user_id,
            session_id=session_id
        )

    def save_interaction(self, user_id: str, session_id: str, query: str, answer: str) -> None:
        self.register_session(user_id=user_id, session_id=session_id)
        importance = self._get_importance(query, answer)
        if importance <= 0:
            return

        self.memory.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query=query,
            answer=answer,
            importance_score=importance
        )

    def save_summary(self, user_id: str, session_id: str, summary: str, facts: str) -> None:
        self.register_session(user_id=user_id, session_id=session_id)
        self.memory.save_summary(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            facts=facts
        )

    def should_store(self, query: str, answer: str) -> bool:
        q = query.strip().lower()
        if q.isdigit():
            return False

        if any(op in q for op in ["+", "-", "*", "/"]):
            return False

        if "could not find" in answer.lower():
            return False

        if "invalid calculation" in answer.lower():
            return False

        return True

    def get_relevant_memory(self, query, user_id, session_id):
        return self.memory.get_relevant_interactions(
            user_id=user_id,
            session_id=session_id,
            query=query
        )

    def _get_importance(self, query: str, answer: str) -> int:
        q = query.lower()
        if any(op in q for op in ["+", "-", "*", "/"]):
            return 0

        if "could not find" in answer.lower():
            return 0

        return 2

    def build_context(self, user_id: str, session_id: str, limit: int = 5) -> str:
        interactions = self.memory.get_recent_interactions(user_id=user_id, session_id=session_id, limit=limit)
        context = ""
        for query, answer in reversed(interactions):
            context += (
                f"User: {query}\n"
                f"Assistant: {answer}\n\n"
            )

        return context
        