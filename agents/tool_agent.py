from utils.logger import logger
from llm.base_llm import BaseLLM
from tools.tool_registry import ToolRegistry
from exceptions.custom_errors import ToolExecutionError


class ToolAgent:
    def __init__(self, registry: ToolRegistry, llm: BaseLLM):
        self.registry = registry
        self.llm = llm

    def execute(self, tool_name: str, query: str) -> str:
        tool = self.registry.get(tool_name)
        logger.info(f"Executing tool: '{tool_name}'")
        return tool(query, self.llm)
