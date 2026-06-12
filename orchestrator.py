from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from agents.tool_agent import ToolAgent
from agents.react_agent import ReActAgent
from agents.reasoning_agent import ReasoningAgent
from agents.classifier import classify_query, QueryCategory
from utils.parallel_runner import run_parallel
from utils.logger import logger


class AgentOrchestrator:

    def __init__(
        self,
        retriever,
        memory_manager,
        reasoning_agent: ReasoningAgent,
        tool_agent: ToolAgent,
        react_agent: ReActAgent,
        llm: BaseLLM,
        llm_config: LLMConfig,
    ):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.reasoning_agent = reasoning_agent
        self.tool_agent = tool_agent
        self.react_agent = react_agent
        self.llm = llm
        self.llm_config = llm_config

    def get_response(self, query: str, user_id: str, session_id: str) -> str:
        category = classify_query(query, self.llm)
        logger.info(f"Route: {category.value} | Query: '{query}'")

        if category == QueryCategory.MATH:
            return self.tool_agent.execute("calculator", query)

        if category == QueryCategory.ACTION:
            return "I can answer questions based on indexed documents. Write/create operations are not supported yet."

        if category == QueryCategory.ANALYTICAL:
            history = self.memory_manager.get_relevant_memory(
                query=query, user_id=user_id, session_id=session_id
            )
            return self.react_agent.run(
                query=query, user_id=user_id, session_id=session_id,
                history=history, config=self.llm_config,
            )

        results = run_parallel([
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