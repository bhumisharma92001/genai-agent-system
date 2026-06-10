from utils.logger import logger
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from exceptions.custom_errors import ReasoningGenerationError, InvalidQueryError
from prompts.reasoning_prompts import get_reasoning_system_prompt

class ReasoningAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _format_context(self, chunks: list[dict]) -> str:
        """Constructs a clean context string for the LLM."""
        extracted = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("page_content") or str(chunk)
            meta = chunk.get("metadata", {})
            header = f"[Source: {meta.get('source', 'Unknown')}, Page: {meta.get('page_number', '')}]"
            extracted.append(f"{header}\n{text}")
        return "\n\n".join(extracted)

    def answer(self, query: str, chunks: list[dict], history: list[tuple[str, str]], config: LLMConfig) -> str:
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty.")
            
        if not chunks:
            return "I could not find the answer in the provided documents."
            
        try:
            context = self._format_context(chunks)
            history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history) if history else ""            
            system_prompt = get_reasoning_system_prompt(context, history_text)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}"}
            ]
            
            response = self.llm.generate(messages=messages, config=config)
            
            if not response:
                raise ReasoningGenerationError("LLM returned empty response.")
                
            return response.strip()

        except InvalidQueryError:
            raise
        except Exception as e:
            logger.error(f"ReasoningAgent critical failure: {str(e)}")
            raise ReasoningGenerationError(f"Generation process failed: {e}") from e