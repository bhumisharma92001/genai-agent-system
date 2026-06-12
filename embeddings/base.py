from abc import ABC, abstractmethod

class BaseEmbedding(ABC):

    @abstractmethod
    def embed(self, text: str, is_query: bool = False)-> list[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        pass