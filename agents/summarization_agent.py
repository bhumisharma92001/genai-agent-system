from utils.logger import logger
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from exceptions.custom_errors import SummarizationError
from prompts.summarization_prompts import get_summarization_prompt

class SummarizationAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.config = LLMConfig(temperature=0.2, top_p=0.8, max_tokens=256)

    def summarize(self, conversation: str) -> str:
        if not conversation or not conversation.strip():
            return ""
        
        try:
            messages = [
                {"role": "system", "content": "You are a professional summarization assistant."},
                {"role": "user", "content": get_summarization_prompt(conversation)}
            ]
            
            response = self.llm.generate(messages=messages, config=self.config)
            
            if not response or not response.strip():
                raise SummarizationError("LLM returned empty summary.")
                
            return response.strip()
        except SummarizationError:
            raise    
        except Exception as e:
            logger.error(f"SummarizationAgent critical fault: {str(e)}")
            raise SummarizationError(f"Summarization process failed: {e}") from e