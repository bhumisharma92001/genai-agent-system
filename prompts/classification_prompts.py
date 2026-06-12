import re

PURE_MATH_RE = re.compile(
    r'^\s*[\d\s\+\-\*/\(\)\.]+\s*$|'
    r'\b\d+\s*[\+\-\*\/]\s*\d+\b|'
    r'\b(average|sum|total)\s+of\s+[\d\s,]+$',
    re.I,
)

def get_classification_prompt(query: str) -> str:
    return (
        "Classify this query into exactly one category: "
        "INFORMATIONAL, ANALYTICAL, MATH, or ACTION.\n\n"
        "MATH: pure arithmetic using ONLY literal numbers already written in the query. "
        "A query is MATH only if it can be solved without looking up any document. "
        "If any value needs to be fetched first — brand, product, entity, or percentage of something — it is ANALYTICAL not MATH.\n"
        "ANALYTICAL: requires searching, listing, filtering, or aggregating data "
        "from documents — any query that needs to look up multiple records, "
        "compare values, summarize data across rows, or look up data by a specific ID, name, or attribute filter.\n"
        "ACTION: requests to create, modify, delete, send, or generate something.\n"
        "INFORMATIONAL: a specific factual question about one known entity "
        "that can be answered from a single document lookup.\n\n"
        "Reply with ONLY the single category word, nothing else.\n"
        f"Query: {query}"
    )