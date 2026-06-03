from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    def execute(self, params: dict) -> dict:
        pass