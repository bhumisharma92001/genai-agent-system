import ast
import re
import operator
from tools.base_tool import BaseTool

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}

def _safe_eval(expression: str):
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported operation")

    return _eval(tree)


class CalculatorTool(BaseTool):

    def execute(self, params: dict) -> dict:
        try:
            query = params["query"].lower().strip()

            replacements = [
                ("plus", "+"), ("minus", "-"),
                ("multiplied by", "*"), ("multiply", "*"), ("times", "*"),
                ("divided by", "/"), ("divide", "/"),
                ("add", "+"), ("and", "+"),
            ]
            for word, symbol in replacements:
                query = query.replace(word, symbol)

            expression = "".join(re.findall(r'[\d\+\-\*/\(\)\.\s]+', query))
            result = _safe_eval(expression)
            return {"result": result}

        except Exception:
            return {"result": "Invalid calculation"}