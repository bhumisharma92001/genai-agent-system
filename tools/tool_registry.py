from exceptions.custom_errors import ToolExecutionError


class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, fn) -> None:
        self._tools[name] = fn

    def get(self, name: str):
        tool = self._tools.get(name)
        if not tool:
            raise ToolExecutionError(f"Tool '{name}' not found in registry.")
        return tool
