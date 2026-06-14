import re


class InteractionFilter:

    CALCULATOR_PATTERN = re.compile(
        r'^[\d\s+\-*/().=]+$|'
        r'\b(add|subtract|multiply|divide|calc|calculate|compute)\b|'
        r'\b(divided\s+by|multiplied\s+with|sum\s+of|product\s+of)\b|'
        r'[\d.]+\s*[+\-*/]\s*[\d.]+',
        re.IGNORECASE
    )

    LOW_INFO_PHRASES = [
        "could not find", "i don't know", "i am sorry", "i'm sorry"
    ]

    def get_importance(self, query: str, answer: str) -> int:
        if self.CALCULATOR_PATTERN.search(query.strip()):
            return 0
        if any(phrase in answer.lower() for phrase in self.LOW_INFO_PHRASES):
            return 0
        return 1

    def should_store(self, query: str, answer: str) -> bool:
        return self.get_importance(query, answer) > 0