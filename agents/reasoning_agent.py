from utils.logger import logger
from exceptions.custom_errors import ReasoningGenerationError, InvalidQueryError
from utils.conversation import build_history
from prompts.reasoning_prompts import get_reasoning_system_prompt
from langchain_core.messages import HumanMessage, SystemMessage


class ReasoningAgent:

    def __init__(self, llm):
        self.llm = llm 

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Build a rich context string from retrieved chunks.
        Table chunks get a structured label so LLM knows they contain data rows.
        """
        parts = []
        for c in chunks:
            source = c.get("metadata", {}).get("source", "Unknown")
            chunk_type = c.get("metadata", {}).get("chunk_type", "text")
            text = c.get("text", str(c))

            if chunk_type in ("table", "table_row"):
                parts.append(f"[TABLE DATA — Source: {source}]\n{text}")
            else:
                parts.append(f"[Source: {source}]\n{text}")

        return "\n\n".join(parts)

    def _messages(self, query: str, chunks: list[dict], history: list[tuple]) -> list:
        context = self._build_context(chunks)
        history_text = build_history(history)
        return [
            SystemMessage(content=get_reasoning_system_prompt(context, history_text)),
            HumanMessage(content=f"Question: {query}"),
        ]

    def answer(self, query, chunks, history):
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty.")
        if not chunks:
            return "I could not find the answer in the provided documents."
        try:
            resp = self.llm.invoke(self._messages(query, chunks, history))
            if not resp.content:
                raise ReasoningGenerationError("LLM returned empty response.")
            return resp.content.strip()
        except (InvalidQueryError, ReasoningGenerationError):
            raise
        except Exception as e:
            logger.error(f"ReasoningAgent failed: {e}")
            raise ReasoningGenerationError(f"Generation failed: {e}") from e