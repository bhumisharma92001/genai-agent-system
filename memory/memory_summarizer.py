import threading
from utils.logger import logger
from memory.base_memory import BaseMemory
from exceptions.custom_errors import EpisodicMemoryError, SummarizationError


class MemorySummarizer:

    def __init__(self, summarization_agent, memory: BaseMemory):
        self.agent = summarization_agent
        self.memory = memory
        self._lock = threading.Lock()
        self._is_summarizing = False
        self._summary_thread: threading.Thread | None = None

    def run(self, user_id: str, session_id: str, blocking: bool = False):
        if blocking:
            if self._summary_thread and self._summary_thread.is_alive():
                logger.info("Joining in-flight background summary thread.")
                self._summary_thread.join()

            with self._lock:
                if self._is_summarizing:
                    logger.info("Blocking summarization already running. Skipping.")
                    return
                self._is_summarizing = True
            try:
                self._process(user_id, session_id)
            finally:
                with self._lock:
                    self._is_summarizing = False
            return

        with self._lock:
            if self._is_summarizing:
                logger.info("Summarization already in progress. Skipping.")
                return
            self._is_summarizing = True

        self._summary_thread = threading.Thread(
            target=self._process,
            args=(user_id, session_id),
            daemon=True
        )
        try:
            self._summary_thread.start()
        except Exception as e:
            with self._lock:
                self._is_summarizing = False
            logger.error(f"Failed to start summarization thread: {e}")
            raise

    def _process(self, user_id: str, session_id: str):
        try:
            existing_summaries = self.memory.get_summaries(
                user_id=user_id, session_id=session_id
            )

            if existing_summaries:
                latest_summary_text, last_summarized_id = existing_summaries[0]
                interactions = self.memory.get_interactions_after_id(
                    user_id=user_id, session_id=session_id, last_id=last_summarized_id
                )
            else:
                latest_summary_text = None
                last_summarized_id = 0
                interactions = self.memory.get_recent_interactions(
                    user_id=user_id, session_id=session_id, limit=15
                )

            if not interactions:
                logger.info("No new interactions found to summarize.")
                return

            conversation = "\n".join(
                f"User: {q}\nAssistant: {a}" for q, a in interactions
            )

            summary = (
                self.agent.summarize_update(latest_summary_text, conversation)
                if latest_summary_text
                else self.agent.summarize_fresh(conversation)
            )

            if summary and summary.strip():
                new_last_id = self.memory.get_max_interaction_id(
                    user_id=user_id, session_id=session_id
                )
                self.memory.save_summary(
                    user_id=user_id, session_id=session_id,
                    summary=summary.strip(),
                    last_summarized_id=new_last_id
                )
                logger.info("New consolidated summary saved successfully.")

        except (EpisodicMemoryError, SummarizationError):
            logger.exception("Known failure in summarizer engine.")
        except Exception:
            logger.exception("Unexpected error in summarizer engine.")
        finally:
            with self._lock:
                self._is_summarizing = False