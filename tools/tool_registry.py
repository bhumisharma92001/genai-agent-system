from tools.calculator_tool import CalculatorTool

class ToolRegistry:

    def __init__(self):
        self.tools = { "calculator": CalculatorTool()}

    def get_tool(self, name: str):
        return self.tools.get(name)