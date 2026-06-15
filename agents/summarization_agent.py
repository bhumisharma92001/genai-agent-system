from utils.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage
from exceptions.custom_errors import SummarizationError
from prompts.summarization_prompts import get_fresh_prompt, get_update_prompt


class SummarizationAgent:

    def __init__(self, llm):
        self.llm = llm
    def _call_llm(self, prompt: str) -> str:
        messages = [
            SystemMessage(content="You are a professional summarization assistant."),
            HumanMessage(content=prompt)]
        resp = self.llm.invoke(messages)
        if not resp.content or not resp.content.strip():
            raise SummarizationError("LLM returned empty summary.")
        return resp.content.strip()

    def summarize_fresh(self, conversation: str) -> str:
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
        if not new_conversation or not new_conversation.strip():
            return existing_summary
        try:
            return self._call_llm(get_update_prompt(existing_summary, new_conversation))
        except SummarizationError:
            raise
        except Exception as e:
            logger.error(f"SummarizationAgent update fault: {e}")
            raise SummarizationError(f"Update summarization failed: {e}") from e