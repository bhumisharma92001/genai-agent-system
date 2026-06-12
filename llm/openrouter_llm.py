from openai import OpenAI
from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from utils.logger import logger
from exceptions.custom_errors import LLMGenerationError, InvalidInputError

class OpenRouterLLM(BaseLLM):

    def __init__(self, api_key: str, model_name: str):
        if not api_key or not model_name:
            raise InvalidInputError("api_key and model_name are required")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name

    def _base_params(self, messages: list[dict], config: LLMConfig) -> dict:
        return dict(model=self.model_name, messages=messages,
                    temperature=config.temperature, top_p=config.top_p,
                    max_tokens=config.max_tokens)

    def generate(self, messages: list[dict], config: LLMConfig) -> str:
        try:
            resp = self.client.chat.completions.create(**self._base_params(messages, config))
            if resp.choices[0].finish_reason != "stop":
                logger.warning(f"finish_reason: {resp.choices[0].finish_reason}")
            content = resp.choices[0].message.content
            if not content:
                raise LLMGenerationError("Empty response from LLM")
            return content
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"LLM generate failed: {e}")
            raise LLMGenerationError(f"LLM generation failed: {e}") from e
