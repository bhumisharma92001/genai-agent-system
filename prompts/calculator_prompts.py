def get_math_extraction_prompt(query: str) -> str:
    """
    Prompt for extracting a clean arithmetic expression from natural language.
    Used when the direct sanitize + eval path fails (e.g. commas, percent words,
    spelled-out numbers). LLM converts the query to a pure math expression
    that _safe_eval can handle.
    """
    return (
        "Extract only the arithmetic expression from this query as digits and operators only.\n"
        "Examples:\n"
        "'what is ninety thousand and twenty five' -> '90000 + 25'\n"
        "'what is 10 percent of 500' -> '500 * 0.10'\n"
        "'sum of 100 and 200' -> '100 + 200'\n"
        "'average of 56, 87, 77' -> '(56 + 87 + 77) / 3'\n"
        "'2 + 2' -> '2 + 2'\n"
        f"Query: {query}\n"
        "Reply with ONLY the math expression, nothing else."
    )
 