from enum import Enum
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from prompts.classification_prompts import get_classification_prompt, PURE_MATH_RE
from utils.logger import logger


class QueryCategory(Enum):
    INFORMATIONAL = "INFORMATIONAL"
    ANALYTICAL = "ANALYTICAL"
    MATH = "MATH"
    ACTION = "ACTION"


def classify_query(query: str, llm: BaseLLM) -> QueryCategory:
    if PURE_MATH_RE.search(query.strip()):
        return QueryCategory.MATH
    resp = llm.generate(
        messages=[{"role": "user", "content": get_classification_prompt(query)}],
        config=LLMConfig(temperature=0, top_p=1.0, max_tokens=10),
    )
    try:
        category = QueryCategory(resp.strip().upper())
        if category == QueryCategory.ACTION:
            logger.warning(f"ACTION query — no handler implemented: '{query}'")
        return category
    except ValueError:
        logger.warning(f"Unknown category '{resp.strip()}'. Defaulting to INFORMATIONAL.")
        return QueryCategory.INFORMATIONAL