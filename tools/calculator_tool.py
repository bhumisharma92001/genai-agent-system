from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    def execute(self, params: dict) -> dict:
        expression = params["expression"]
        result = eval(expression)
        return {"result": result}