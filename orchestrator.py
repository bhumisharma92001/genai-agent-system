import concurrent.futures
from enum import Enum

from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from agents.tool_agent import ToolAgent
from prompts.classification_prompts import get_classification_prompt, MATH_RE, ANALYTICAL_RE, ACTION_RE
from utils.logger import logger


class QueryCategory(Enum):
    INFORMATIONAL = "INFORMATIONAL"
    ANALYTICAL = "ANALYTICAL"
    MATH = "MATH"
    ACTION = "ACTION"


def _classify(query: str, llm: BaseLLM) -> QueryCategory:
    if MATH_RE.search(query):
        return QueryCategory.MATH
    if ANALYTICAL_RE.search(query):
        return QueryCategory.ANALYTICAL
    if ACTION_RE.search(query):
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
    def __init__(self, retriever, memory_manager, reasoning_agent, tool_agent: ToolAgent, llm: BaseLLM, llm_config: LLMConfig):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.reasoning_agent = reasoning_agent
        self.tool_agent = tool_agent
        self.llm = llm
        self.llm_config = llm_config

    def get_response(self, query: str, user_id: str, session_id: str) -> str:
        category = _classify(query, self.llm)
        logger.info(f"Route: {category.value} | Query: '{query}'")

        if category == QueryCategory.MATH:
            tool_result = self.tool_agent.execute("calculator", query)
            return self.reasoning_agent.observe(
                query=query, tool_result=tool_result, config=self.llm_config
            )

        if category == QueryCategory.ACTION:
            return "I can answer questions based on indexed documents. Write/create operations are not supported yet."

        results = _run_parallel([
            {"name": "chunks", "fn": self.retriever.retrieve,
             "args": {"query": query, "user_id": user_id, "session_id": session_id}},
            {"name": "history", "fn": self.memory_manager.get_relevant_memory,
             "args": {"query": query, "user_id": user_id, "session_id": session_id}},
        ])

        chunks = results.get("chunks") or []
        history = results.get("history") or []

        if not chunks:
            return "I could not find the answer in the provided documents."

        return self.reasoning_agent.answer(
            query=query, chunks=chunks, history=history, config=self.llm_config
        )