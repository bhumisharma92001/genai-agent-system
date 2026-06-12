import json
import re
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig


class QueryRouter:

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def classify(self, query: str, config: LLMConfig | None = None) -> dict:

        # ✅ HARD RULE (MATH DETECTION BEFORE LLM)
        if (
            re.search(r"\d+\s*[\+\-\*/]\s*\d+", query)
            or re.search(r"\b(plus|minus|add|subtract|multiply|divide)\b", query.lower())
        ):
            return {
                "route": "tool",
                "tool": "calculator",
                "reason": "rule-based math detection"
            }

        if config is None:
            config = LLMConfig(temperature=0, max_tokens=16)

        messages = [
            {
                "role": "system",
                "content": """
You are an intelligent routing system.

Classify the user query into ONE route:

Routes:
- tool → ONLY for arithmetic, math, calculation, or numeric operations
- rag → if answer exists in documents
- reasoning → general knowledge or explanation

IMPORTANT RULES:
- If query contains ANY math expression like 7+7, 10*5, 100/2 → tool
- If query contains words like add, plus, minus, divide → tool
- Never treat math as rag

Return ONLY valid JSON:
{
  "route": "rag | tool | reasoning",
  "tool": "calculator or null",
  "reason": "short explanation"
}
"""
            },
            {"role": "user", "content": query}
        ]

        response = self.llm.generate(messages=messages, config=config)
        return self._safe_parse(response)

    def _safe_parse(self, text: str) -> dict:
        try:
            payload = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group())
                except Exception:
                    payload = None
            else:
                payload = None

        if not isinstance(payload, dict):
            return {
                "route": "rag",
                "tool": None,
                "reason": "fallback parsing failed"
            }

        route = str(payload.get("route", "rag")).strip().lower()

        route_map = {
            "tool": "tool",
            "calculator": "tool",
            "rag": "rag",
            "reasoning": "reasoning",
        }

        payload["route"] = route_map.get(route, "rag")
        payload["tool"] = payload.get("tool") if payload.get("tool") else None
        payload["reason"] = payload.get("reason", "parsed route")

        return payload