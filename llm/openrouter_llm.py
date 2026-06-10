from openai import OpenAI
from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from utils.logger import logger
from exceptions.custom_errors import LLMGenerationError, InvalidInputError

class OpenRouterLLM(BaseLLM):

    def __init__(self,api_key: str,model_name: str ):
        if not api_key or not api_key.strip():
            raise InvalidInputError("OPENROUTER_API_KEY cannot be empty")
        if not model_name or not model_name.strip():
            raise InvalidInputError("model_name cannot be empty")
        self.client = OpenAI(api_key=api_key,base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name

    def generate(self, messages: list[dict], config: LLMConfig) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens
            )

            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                logger.warning(f"Generation stopped with reason: {finish_reason}")

            content = response.choices[0].message.content
            if not content:
                raise LLMGenerationError("LLM returned empty response")
            return content
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"LLM API Call failed: {str(e)}")
            raise LLMGenerationError(f"LLM generation failed: {e}") from e