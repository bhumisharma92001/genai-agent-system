import re
import concurrent.futures
from enum import Enum

from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from prompts.classification_prompts import get_classification_prompt
from exceptions.custom_errors import OrchestrationError
from utils.logger import logger


class QueryCategory(Enum):
    INFORMATIONAL = "INFORMATIONAL"
    ANALYTICAL = "ANALYTICAL"
    MATH = "MATH"
    ACTION = "ACTION"


_MATH_RE = re.compile(
    r'\b(\d+\s*[\+\-\*/]\s*\d+|'
    r'what\s+is\s+\d+|'
    r'(sum|add|plus)\s+of\s+\d+\s+(and|\+)\s+\d+|'
    r'\d+\s+(plus|minus|times|divided\s+by|multiplied\s+by)\s+\d+)\b',
    re.I,
)
_ANALYTICAL_RE = re.compile(
    r'\b(average|mean|median|trend|compare|total|revenue|salary|'
    r'count|max|min|filter|group\s+by|last\s+\d+|top\s+\d+|per\s+quarter|across)\b',
    re.I,
)
_ACTION_RE = re.compile(
    r'\b(create|update|delete|send|write|generate|make|set|add|remove|edit|post|submit|'
    r'upload|download|save|export|import|schedule|book|cancel|assign|notify)\b',
    re.I,
)


def _classify(query: str, llm: BaseLLM) -> QueryCategory:
    if _MATH_RE.search(query):
        return QueryCategory.MATH
    if _ANALYTICAL_RE.search(query):
        return QueryCategory.ANALYTICAL
    if _ACTION_RE.search(query):
        return QueryCategory.ACTION

    resp = llm.generate(
        messages=[{"role": "user", "content": get_classification_prompt(query)}],
        config=LLMConfig(temperature=0, top_p=1.0, max_tokens=10),
    )
    try:
        return QueryCategory(resp.strip().upper())
    except ValueError:
        logger.warning(f"LLM returned unknown category '{resp.strip()}'. Defaulting to INFORMATIONAL.")
        return QueryCategory.INFORMATIONAL


def _run_parallel(tasks: list[dict]) -> dict:
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(t["fn"], **t["args"]): t["name"] for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception as e:
                logger.error(f"Parallel task '{name}' failed: {e}")
                results[name] = None
    return results


class AgentOrchestrator:
    """
    Deterministic pipeline:
    classify → parallel(retrieve + memory) → generate(reasoning)

    Streaming removed — full response returned at once.
    """

    def __init__(
        self,
        retriever,
        memory_manager,
        reasoning_agent,
        calculator_tool,
        llm: BaseLLM,
        llm_config: LLMConfig,
    ):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.reasoning_agent = reasoning_agent
        self.calculator_tool = calculator_tool
        self.llm = llm
        self.llm_config = llm_config

    def get_response(self, query: str, user_id: str, session_id: str) -> str:
        category = _classify(query, self.llm)
        logger.info(f"Route: {category.value} | Query: '{query}'")

        if category == QueryCategory.MATH:
            return self.calculator_tool(query)

        results = _run_parallel([
            {
                "name": "chunks",
                "fn": self.retriever.retrieve,
                "args": {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                },
            },
            {
                "name": "history",
                "fn": self.memory_manager.get_relevant_memory,
                "args": {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                },
            },
        ])

        chunks = results.get("chunks") or []
        history = results.get("history") or []

        if not chunks:
            return "I could not find the answer in the provided documents."

        messages = self.reasoning_agent._messages(
            query=query, chunks=chunks, history=history
        )
        try:
            return self.llm.generate(messages=messages, config=self.llm_config)
        except Exception as e:
            logger.error(f"Orchestrator generate failed: {e}")
            raise OrchestrationError(f"Generation failed: {e}") from e