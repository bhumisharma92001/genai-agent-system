from memory.base_memory import BaseMemory
class MemoryManager:

    def __init__(
        self,
        memory : BaseMemory
    ):
        self.memory = memory

    def save_interaction(
        self,
        query: str,
        answer: str
    ) -> None:

        self.memory.save_interaction(
            query=query,
            answer=answer
        )

    def get_recent_interactions(
        self,
        limit: int = 5
    )-> list:

        return self.memory.get_recent_interactions(
            limit=limit
        )

    def get_summaries(
        self
    ):

        return self.memory.get_summaries()

    def save_summary(
        self,
        summary: str,
        facts: str
    ) -> None:

        self.memory.save_summary(
            summary=summary,
            facts=facts
        )

    def build_context(
        self,
        limit: int = 5
    ) -> str:

        interactions = (
            self.memory.get_recent_interactions(
                limit=limit
            )
        )

        context = ""

        for query, answer in reversed(interactions):

            context += (
                f"User: {query}\n"
                f"Assistant: {answer}\n\n"
            )

        return context