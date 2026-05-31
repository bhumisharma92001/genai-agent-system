from openai import OpenAI

from llm.base_llm import BaseLLM


class OpenRouterLLM(BaseLLM):

    def __init__(
        self,
        api_key: str,
        model_name: str = "google/gemini-2.5-flash"
    ):

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        config
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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