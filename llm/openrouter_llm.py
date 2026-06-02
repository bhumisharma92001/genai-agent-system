from openai import OpenAI
from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM

class OpenRouterLLM(BaseLLM):

    def __init__(self,api_key: str,model_name: str ):
        if not api_key or not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY not found")

        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self.client = OpenAI(api_key=api_key,base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name

    def generate(self,messages: list[dict],config : LLMConfig) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens
        )

        return (
            response
            .choices[0]
            .message
            .content
        )