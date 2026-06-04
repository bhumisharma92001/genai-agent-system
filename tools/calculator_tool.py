import re

from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):

    def execute(self, params: dict) -> dict:

        query = params["query"].lower().strip()

        query = query.replace("plus", "+")
        query = query.replace("minus", "-")

        query = query.replace("multiplied by", "*")
        query = query.replace("multiply", "*")
        query = query.replace("times", "*")

        query = query.replace("divided by", "/")
        query = query.replace("divide", "/")
        query = query.replace("add", "+")
        query = query.replace("and", "+")

        query = query.replace("by", "")

        if "=" in query:
            left, right = query.split("=")
            left_expression = "".join(re.findall(r'[\d\+\-\*/\(\)\.\s]+', left))
            right_expression = "".join(re.findall(r'[\d\+\-\*/\(\)\.\s]+', right))
            return {"result": eval(left_expression) == eval(right_expression)}

        expression = "".join(re.findall(r'[\d\+\-\*/\(\)\.\s]+',query))

        try:
            result = eval(expression)
            return {"result": result}

        except Exception:
            return {"result": "Invalid calculation"}