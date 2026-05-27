from abc import ABC, abstractmethod


class BaseVectorDB(ABC):

    @abstractmethod
    def upsert(self, vectors, metadata):
        pass

    @abstractmethod
    def query(self, vector, k):
        pass