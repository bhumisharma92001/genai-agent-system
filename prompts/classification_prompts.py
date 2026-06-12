import re

MATH_RE = re.compile(
    r'\b(\d+\s*[\+\-\*/]\s*\d+|'
    r'what\s+is\s+\d+|'
    r'(sum|add|plus)\s+(of\s+)?\d+\s+(and|\+)\s+\d+|'
    r'\d+\s+(plus|minus|times|divided\s+by|multiplied\s+by)\s+\d+)\b',
    re.I,
)
ANALYTICAL_RE = re.compile(
    r'\b(average|mean|median|trend|compare|total|revenue|salary|'
    r'count|max|min|filter|group\s+by|last\s+\d+|top\s+\d+|per\s+quarter|across)\b',
    re.I,
)
ACTION_RE = re.compile(
    r'\b(create|update|delete|send|write|generate|make|set|add|remove|edit|post|submit|'
    r'upload|download|save|export|import|schedule|book|cancel|assign|notify)\b',
    re.I,
)

def get_classification_prompt(query: str) -> str:
    return (
        "Classify this query into exactly one category: "
        "INFORMATIONAL, ANALYTICAL, MATH, or ACTION.\n\n"
        "MATH: pure arithmetic only (e.g. 2+2, what is 10% of 500).\n"
        "ANALYTICAL: aggregations or analysis on document data "
        "(e.g. average salary, total revenue, compare trends).\n"
        "ACTION: requests to create, update, delete, send, generate, "
        "schedule, or perform any write/mutation operation "
        "(e.g. write a summary, create a report, send an email).\n"
        "INFORMATIONAL: everything else that requires document context "
        "to answer a question.\n\n"
        "Reply with ONLY the single category word, nothing else.\n"
        f"Query: {query}"
    )