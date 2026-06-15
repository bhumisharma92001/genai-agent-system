from enum import Enum
from langchain_core.messages import HumanMessage
from prompts.classification_prompts import get_classification_prompt, PURE_MATH_RE
from utils.logger import logger


class QueryCategory(Enum):
    INFORMATIONAL = "INFORMATIONAL"
    ANALYTICAL = "ANALYTICAL"
    MATH = "MATH"
    ACTION = "ACTION"


def classify_query(query: str, llm) -> QueryCategory:
    if PURE_MATH_RE.search(query.strip()):
        return QueryCategory.MATH
    resp = llm.invoke([HumanMessage(content=get_classification_prompt(query))])
    try:
        category = QueryCategory(resp.content.strip().upper())
        return category
    except ValueError:
        logger.warning(f"Unknown category '{resp.content.strip()}'.")
        return QueryCategory.INFORMATIONAL