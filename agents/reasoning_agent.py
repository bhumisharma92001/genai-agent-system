from utils.logger import logger
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from exceptions.custom_errors import ReasoningGenerationError, InvalidQueryError
from prompts.reasoning_prompts import get_reasoning_system_prompt


class ReasoningAgent:

    def __init__(self, llm: BaseLLM):
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

    def _build_history(self, history: list[tuple]) -> str:
        """
        Build conversation history string.
        Summary tuple alag block mein render hoti hai.
        Regular interactions User/Assistant format mein.
        """
        if not history:
            return ""
        lines = []
        for q, a in history:
            if q == "Previous conversation summary":
                lines.append(f"[CONVERSATION SUMMARY]\n{a}")
                lines.append("")
            else:
                lines.append(f"User: {q}")
                lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def _messages(self, query: str, chunks: list[dict], history: list[tuple]) -> list[dict]:
        context = self._build_context(chunks)
        history_text = self._build_history(history)
        return [
            {"role": "system", "content": get_reasoning_system_prompt(context, history_text)},
            {"role": "user", "content": f"Question: {query}"},
        ]

    def answer(self, query: str, chunks: list[dict], history: list[tuple], config: LLMConfig) -> str:
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty.")
        if not chunks:
            return "I could not find the answer in the provided documents."
        try:
            resp = self.llm.generate(messages=self._messages(query, chunks, history), config=config)
            if not resp:
                raise ReasoningGenerationError("LLM returned empty response.")
            return resp.strip()
        except (InvalidQueryError, ReasoningGenerationError):
            raise
        except Exception as e:
            logger.error(f"ReasoningAgent failed: {e}")
            raise ReasoningGenerationError(f"Generation failed: {e}") from e

    def observe(self, query: str, tool_result: str, config: LLMConfig) -> str:
        """Tool result ko LLM se natural language mein explain karwao."""
        if not tool_result or not tool_result.strip():
            return "I could not compute the result."
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Explain the result of a calculation clearly and naturally."},
                {"role": "user", "content": f"Question: {query}\nCalculation Result: {tool_result}\nExplain this result in a natural, helpful way."}
            ]
            resp = self.llm.generate(messages=messages, config=config)
            if not resp:
                return tool_result
            return resp.strip()
        except Exception as e:
            logger.error(f"ReasoningAgent observe failed: {e}")
            return tool_result