import google.generativeai as genai

from llm.base_llm import BaseLLM


class GeminiLLM(BaseLLM):

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash"
    ):

        if not api_key.strip():

            raise ValueError(
                "api_key cannot be empty"
            )

        try:

            genai.configure(
                api_key=api_key
            )

            self.model = (
                genai.GenerativeModel(
                    model_name
                )
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to initialize Gemini"
            ) from e

    def generate(
        self,
        prompt: str,
        config
    ) -> str:

        if not prompt.strip():

            raise ValueError(
                "prompt cannot be empty"
            )

        try:

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature":
                        config.temperature,
                    "top_p":
                        config.top_p,
                    "max_output_tokens":
                        config.max_tokens
                }
            )

            return response.text

        except Exception as e:

            raise RuntimeError(
                "Failed to generate response"
            ) from e