from abc import ABC, abstractmethod
from collections.abc import Iterator
from llm.llm_config import LLMConfig

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], config: LLMConfig) -> str:
        pass

    @abstractmethod
    def stream(self, messages: list[dict], config: LLMConfig) -> Iterator[str]:
        pass