from llm.base_llm import BaseLLM
from tools.tool_registry import ToolRegistry

class ToolAgent:

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.registry = ToolRegistry()

    def execute(self, query: str):
        calculator = self.registry.get_tool("calculator")
        return calculator.execute({"query": query})