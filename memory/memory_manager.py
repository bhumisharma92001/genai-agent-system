from memory.base_memory import BaseMemory
import threading
from utils.logger import logger

class MemoryManager:

    def __init__(self, memory: BaseMemory, summarization_agent):
        self.memory = memory
        self.summarization_agent = summarization_agent

    def summarize_in_background(self):
        thread = threading.Thread(target=self._generate_summary,daemon=True)
        thread.start()

    def _generate_summary(self):
        try:
            interactions = self.memory.get_recent_interactions(limit=3)
            conversation = "\n".join(
                f"User: {query}\nAssistant: {answer}"
                for query, answer in interactions
            )

            summary = self.summarization_agent.summarize(conversation)

            self.memory.save_summary(summary=summary,facts="")

        except Exception:
            logger.exception("Background summarizer failed")

    def save_interaction(self,query: str,answer: str) -> None:
        self.memory.save_interaction(query=query,answer=answer)

    def get_recent_interactions(self,limit: int = 5)-> list:
        return self.memory.get_recent_interactions(limit=limit)

    def get_summaries(self)-> list:
        return self.memory.get_summaries()

    def save_summary(self,summary: str,facts: str) -> None:
        self.memory.save_summary(summary=summary,facts=facts)

    def build_context(self,limit: int = 5) -> str:
        interactions = (self.memory.get_recent_interactions(limit=limit))
        context = ""
        for query, answer in reversed(interactions):
            context += (
                f"User: {query}\n"
                f"Assistant: {answer}\n\n"
            )

        return context
        