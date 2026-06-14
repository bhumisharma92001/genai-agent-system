from llm.llm_config import LLMConfig
from llm.base_llm import BaseLLM
from agents.react_agent import ReActAgent
from agents.reasoning_agent import ReasoningAgent
from agents.classifier import classify_query, QueryCategory
from utils.parallel_runner import run_parallel
from utils.logger import logger
from exceptions.custom_errors import OrchestrationError
from tools.tool_registry import ToolRegistry


class AgentOrchestrator:

    def __init__(
        self,
        retriever,
        memory_manager,
        reasoning_agent: ReasoningAgent,
        registry: ToolRegistry,        # ← clean param
        react_agent: ReActAgent,
        llm: BaseLLM,
        llm_config: LLMConfig,
    ):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.reasoning_agent = reasoning_agent
        self.registry = registry
        self.react_agent = react_agent
        self.llm = llm
        self.llm_config = llm_config

    def get_response(self, query: str, user_id: str, session_id: str) -> str:
        try:
            category = classify_query(query, self.llm)
            logger.info(f"Route: {category.value} | Query: '{query}'")

            if category == QueryCategory.MATH:
                return self.registry.get("calculator")(query)

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

            if results.get("chunks") is None:
                logger.error("Retriever task failed in parallel execution.")
                return "I encountered an issue processing your request. Please try again."
            chunks = results.get("chunks") or []
            history = results.get("history") or []

            if not chunks:
                return "I could not find the answer in the provided documents."

            return self.reasoning_agent.answer(
                query=query, chunks=chunks, history=history, config=self.llm_config
            )

        except OrchestrationError:
            raise
        except Exception as e:
            logger.exception(f"Orchestrator failed for query: '{query}'")
            raise OrchestrationError(f"Failed to process query: {e}") from e