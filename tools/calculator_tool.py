import ast
import operator
import re
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from exceptions.custom_errors import ToolExecutionError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
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

_MATH_EXTRACT_CONFIG = LLMConfig(temperature=0, top_p=1.0, max_tokens=20)


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


def _extract_math_expression(query: str, llm: BaseLLM) -> str:
    """LLM se clean math expression nikalo natural language se."""
    resp = llm.generate(
        messages=[{"role": "user", "content":
            f"Extract only the arithmetic expression from this query as digits and operators only.\n"
            f"Examples:\n"
            f"'what is ninety thousand and twenty five' → '90000 + 25'\n"
            f"'what is 10 percent of 500' → '500 * 0.10'\n"
            f"'sum of 100 and 200' → '100 + 200'\n"
            f"'2 + 2' → '2 + 2'\n"
            f"Query: {query}\n"
            f"Reply with ONLY the math expression, nothing else."
        }],
        config=_MATH_EXTRACT_CONFIG,
    )
    return resp.strip()


def calculator(query: str, llm: BaseLLM) -> str:
    """Evaluate arithmetic expressions from natural language."""
    try:
        # Step 1 — pehle seedha try karo without LLM
        expr = query.strip().split("=")[0].strip()
        for pattern, symbol in _REPLACEMENTS:
            expr = re.sub(pattern, symbol, expr)
        clean = "".join(re.findall(r"[\d+\-*/().\s]+", expr))

        if clean.strip():
            try:
                result = _safe_eval(clean)
                return str(round(float(result), 6)).rstrip("0").rstrip(".")
            except ToolExecutionError:
                pass  # Direct eval failed — LLM try karo

        # Step 2 — sirf tab LLM use karo jab direct eval fail ho
        expr = _extract_math_expression(query, llm)
        expr = expr.split("=")[0].strip()
        for pattern, symbol in _REPLACEMENTS:
            expr = re.sub(pattern, symbol, expr)
        clean = "".join(re.findall(r"[\d+\-*/().\s]+", expr))
        if not clean.strip():
            return "Calculation error: No valid expression found in query."

        result = _safe_eval(clean)
        return str(round(float(result), 6)).rstrip("0").rstrip(".")

    except ToolExecutionError as e:
        return f"Calculation error: {e}"
    except Exception as e:
        return f"Calculation error: Unexpected failure — {e}"
