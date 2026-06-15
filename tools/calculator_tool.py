import ast
import operator
import re
from exceptions.custom_errors import ToolExecutionError
from prompts.calculator_prompts import get_math_extraction_prompt
from langchain_core.messages import HumanMessage

_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,    
    ast.USub: operator.neg,
}

_REPLACEMENTS = [
    (r'\bmultiplied\s+by\b', '*'),
    (r'\bdivided\s+by\b', '/'),
    (r'\bplus\b', '+'),
    (r'\bminus\b', '-'),
    (r'\btimes\b', '*'),
    (r'\badd\b', '+'),
    (r'\band\b', '+'),
]



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


def _sanitize_expr(expr: str) -> str:
    """
    Remove characters that are not valid in a math expression.

    Fix C-3: replaces the old re.findall + ''.join() approach which split
    on commas and concatenated fragments — "56,87,77" became "568777".
    Uses re.sub to DELETE invalid chars instead. If the result is
    non-evaluable (e.g. "56 87 77" — no operator), _safe_eval raises
    and the call falls through to the LLM path correctly.
    """
    cleaned = re.sub(r"[^\d+\-*/().\s]", " ", expr)
    return re.sub(r"\s+", " ", cleaned).strip()


def _format_result(value: float) -> str:
    """
    Format a numeric result without trailing decimal zeros.

    Fix: always convert via :.6f before stripping.
    The old approach — str(round(v, 6)).rstrip('0') — failed when _safe_eval
    returned an int (e.g. 500): str(500).rstrip('0') = '5'.
    Using :.6f forces '500.000000' -> rstrip('0') -> '500.' -> rstrip('.') -> '500'.
    """
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _extract_math_expression(query: str, llm) -> str:
    resp = llm.invoke([HumanMessage(content=get_math_extraction_prompt(query))])
    return resp.content.strip()


def calculator(query: str,llm) -> str:
    """Evaluate arithmetic expressions from natural language."""
    try:
        expr = query.strip().split("=")[0].strip()
        for pattern, symbol in _REPLACEMENTS:
            expr = re.sub(pattern, symbol, expr, flags=re.I)
        clean = _sanitize_expr(expr)

        if clean:
            try:
                result = _safe_eval(clean)
                return _format_result(result)
            except ToolExecutionError:
                pass 

        expr = _extract_math_expression(query,llm)
        expr = expr.split("=")[0].strip()
        for pattern, symbol in _REPLACEMENTS:
            expr = re.sub(pattern, symbol, expr, flags=re.I)
        clean = _sanitize_expr(expr)

        if not clean:
            return "Calculation error: No valid expression found in query."

        result = _safe_eval(clean)
        return _format_result(result)

    except ToolExecutionError as e:
        return f"Calculation error: {e}"
    except Exception as e:
        return f"Calculation error: Unexpected failure — {e}"