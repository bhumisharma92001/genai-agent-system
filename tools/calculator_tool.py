import ast
import operator as op
import re

from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):

    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
    }

    def execute(self, params: dict) -> dict:
        query = params["query"].lower().strip()
        expression = self._normalize_query(query)

        try:
            self._validate_expression(expression)
            value = self._evaluate(expression)
            return {"result": value}
        except Exception:
            return {"result": "Invalid calculation"}

    def _normalize_query(self, query: str) -> str:
        replacements = {
            "plus": "+",
            "minus": "-",
            "multiplied by": "*",
            "multiply": "*",
            "times": "*",
            "divided by": "/",
            "divide": "/",
            "add": "+",
            "and": "+",
        }

        for token, symbol in replacements.items():
            query = query.replace(token, symbol)

        expression = "".join(re.findall(r"[\d\+\-\*/\(\)\.\s]+", query))
        return expression.strip()

    def _validate_expression(self, expression: str) -> None:
        if not expression:
            raise ValueError("Empty expression")

        invalid = re.search(r"[^\d\+\-\*/\(\)\.\s]", expression)
        if invalid:
            raise ValueError("Invalid characters in expression")

    def _evaluate(self, expression: str):
        node = ast.parse(expression, mode="eval").body
        return self._eval_node(node)

    def _eval_node(self, node):
        if isinstance(node, ast.BinOp):
            operator = self.operators.get(type(node.op))
            if operator is None:
                raise ValueError("Unsupported operator")
            return operator(
                self._eval_node(node.left),
                self._eval_node(node.right),
            )

        if isinstance(node, ast.Constant):
            return node.value

        raise ValueError("Unsupported expression")