import re
from tools.tool_registry import ToolRegistry

class ToolAgent:
    def __init__(self):
        self.registry = ToolRegistry()

    def should_use_tool(self, query: str) -> bool:
        query = query.strip()
        return bool(re.fullmatch(r'[\d\+\-\*/\(\)\.\s]+',query))

    def execute(self, query: str):
        calculator = self.registry.get_tool("calculator")

        return calculator.execute({"expression": query})