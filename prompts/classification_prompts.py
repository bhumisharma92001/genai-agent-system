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