import os
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model_name:
        raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL must be set")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 512)),
    )