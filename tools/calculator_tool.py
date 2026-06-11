import ast
import operator
import re
from exceptions.custom_errors import ToolExecutionError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ToolExecutionError("Unsupported operation")

    try:
        return _eval(ast.parse(expr.strip(), mode="eval"))
    except ToolExecutionError:
        raise
    except Exception as e:
        raise ToolExecutionError(f"Invalid expression: {e}") from e


def calculator(query: str) -> str:
    """Evaluate arithmetic expressions from natural language."""
    try:
        q = query.lower()
        for word, sym in [
            ("plus", "+"),
            ("minus", "-"),
            ("multiplied by", "*"),
            ("times", "*"),
            ("divided by", "/"),
            ("add", "+"),
            ("and", "+"),
        ]:
            q = q.replace(word, sym)

        expr = "".join(re.findall(r"[\d+\-*/().\s]+", q))
        if not expr.strip():
            return "Calculation error: No valid expression found in query."

        result = _safe_eval(expr)
        formatted = str(round(float(result), 6)).rstrip("0").rstrip(".")
        return formatted

    except ToolExecutionError as e:
        return f"Calculation error: {e}"
    except Exception as e:
        return f"Calculation error: Unexpected failure — {e}"