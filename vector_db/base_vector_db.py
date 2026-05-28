from abc import ABC, abstractmethod


class BaseVectorDB(ABC):

    @abstractmethod
    def upsert(
        self,
        ids,
        embeddings,
        documents,
        metadatas
    ):
        pass

    @abstractmethod
    def query(
        self,
        embedding,
        k: int = 5
    ):
        pass