from utils.logger import logger
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from exceptions.custom_errors import SummarizationError
from prompts.summarization_prompts import get_fresh_prompt, get_update_prompt


class SummarizationAgent:

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.config = LLMConfig(temperature=0.2, top_p=0.8, max_tokens=256)

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a professional summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.generate(messages=messages, config=self.config)
        if not response or not response.strip():
            raise SummarizationError("LLM returned empty summary.")
        return response.strip()

    def summarize(self, prompt: str) -> str:
        """Generic — direct prompt pass karo."""
        if not prompt or not prompt.strip():
            return ""
        try:
            return self._call_llm(prompt)
        except SummarizationError:
            raise
        except Exception as e:
            logger.error(f"SummarizationAgent fault: {e}")
            raise SummarizationError(f"Summarization failed: {e}") from e

    def summarize_fresh(self, conversation: str) -> str:
        """Pehli baar — fresh summary banao."""
        if not conversation or not conversation.strip():
            return ""
        try:
            return self._call_llm(get_fresh_prompt(conversation))
        except SummarizationError:
            raise
        except Exception as e:
            logger.error(f"SummarizationAgent fresh fault: {e}")
            raise SummarizationError(f"Fresh summarization failed: {e}") from e

    def summarize_update(self, existing_summary: str, new_conversation: str) -> str:
        """Existing summary update karo."""
        if not new_conversation or not new_conversation.strip():
            return existing_summary
        try:
            return self._call_llm(get_update_prompt(existing_summary, new_conversation))
        except SummarizationError:
            raise
        except Exception as e:
            logger.error(f"SummarizationAgent update fault: {e}")
            raise SummarizationError(f"Update summarization failed: {e}") from e