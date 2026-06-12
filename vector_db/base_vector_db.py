from abc import ABC, abstractmethod


class BaseVectorDB(ABC):

    @abstractmethod
    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        pass

    @abstractmethod
    def query(self, embedding: list[float], k: int = 5, where: dict | None = None) -> dict:
        pass

    @abstractmethod
    def delete(self, where: dict) -> None:
        pass

    @abstractmethod
    def list_sources(self, where: dict | None = None) -> list[str]:
        pass