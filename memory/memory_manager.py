import re
import threading
from utils.logger import logger

class MemoryManager:
    def __init__(self, memory, summarization_agent):
        self.memory = memory
        self.summarization_agent = summarization_agent
        self._is_summarizing = False
        self._lock = threading.Lock()

    def register_session(self, user_id: str, session_id: str) -> None:
        self.memory.create_or_update_session(user_id=user_id, session_id=session_id)

    def summarize_in_background(self, user_id: str, session_id: str, blocking: bool = False):
        with self._lock:
            if self._is_summarizing:
                logger.info("Summarization already in progress. Skipping execution.")
                return
            self._is_summarizing = True

        if blocking:
            self._generate_summary(user_id, session_id)
        else:
            thread = threading.Thread(
                target=self._generate_summary,
                args=(user_id, session_id),
                daemon=True
            )
            thread.start()

    def _generate_summary(self, user_id: str, session_id: str):
        try:
            existing_summaries = self.memory.get_summaries(user_id=user_id, session_id=session_id)
            
            if existing_summaries:
                latest_summary_text, latest_timestamp = existing_summaries[0]
                interactions = self.memory.get_interactions_after_timestamp(
                    user_id=user_id, session_id=session_id, timestamp=latest_timestamp
                )
            else:
                latest_summary_text = None
                interactions = self.memory.get_recent_interactions(
                    user_id=user_id, session_id=session_id, limit=15
                )
            
            if not interactions:
                logger.info("No new updates found to append to current summary.")
                return
            
            conversation = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in interactions)
            
            if latest_summary_text:
                prompt_context = (
                    f"You are updating a rolling conversation summary.\n"
                    f"Existing Summary:\n{latest_summary_text}\n\n"
                    f"New Distinct Recent Interactions:\n{conversation}\n\n"
                    f"Generate a brand new consolidated summary incorporating both without repeating info."
                )
            else:
                prompt_context = f"Generate a clear, concise summary of this conversation:\n{conversation}"
                
            summary = self.summarization_agent.summarize(prompt_context)
            
            if summary and summary.strip():
                self.memory.save_summary(user_id=user_id, session_id=session_id, summary=summary.strip())
                logger.info("New consolidated summary state written successfully.")
            
        except Exception:
            logger.exception("Summarizer engine execution failed.")
        finally:
            with self._lock:
                self._is_summarizing = False

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = 5) -> list:
        return self.memory.get_recent_interactions(user_id=user_id, session_id=session_id, limit=limit)

    def get_summaries(self, user_id: str, session_id: str) -> list:
        return self.memory.get_summaries(user_id=user_id, session_id=session_id)

    def get_last_session_id(self, user_id: str) -> str | None:
        return self.memory.get_last_session_id(user_id=user_id)

    def save_interaction(self, user_id: str, session_id: str, query: str, answer: str) -> None:
        importance = self._get_importance(query, answer)
        self.memory.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query=query,
            answer=answer,
            importance_score=importance
        )

    def should_store(self, query: str, answer: str) -> bool:
        return self._get_importance(query, answer) > 0

    def get_relevant_memory(self, query: str, user_id: str, session_id: str) -> list:
        interactions = self.memory.get_relevant_interactions(user_id=user_id, session_id=session_id, query=query)
        summaries = self.memory.get_summaries(user_id=user_id, session_id=session_id)
        if summaries:
            summary_turn = ("Previous conversation summary", summaries[0][0])
            return [summary_turn] + interactions
        return interactions

    CALCULATOR_PATTERN = re.compile(
        r'^[\d\s+\-*/().=]+$|'
        r'\b(add|subtract|multiply|divide)\s+\d+|'
        r'\b(sum|product)\s+of\s+\d+',
        re.IGNORECASE
    )

    def _get_importance(self, query: str, answer: str) -> int:
        q = query.strip()
        if self.CALCULATOR_PATTERN.search(q):
            return 0

        low_info_phrases = ["could not find", "i don't know", "i am sorry", "i'm sorry"]
        if any(phrase in answer.lower() for phrase in low_info_phrases):
            return 0

        return 2