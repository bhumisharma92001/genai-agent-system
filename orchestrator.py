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
        llm,
    ):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.reasoning_agent = reasoning_agent
        self.registry = registry
        self.react_agent = react_agent
        self.llm = llm
        self._handlers = {
            QueryCategory.MATH: self._handle_math,
            QueryCategory.ACTION: self._handle_action,
            QueryCategory.ANALYTICAL: self._handle_analytical,
            QueryCategory.INFORMATIONAL: self._handle_informational,
        }

    def _handle_math(self, query: str, user_id: str, session_id: str) -> str:
        return self.registry.get("calculator")(query)

    def _handle_action(self, query: str, user_id: str, session_id: str) -> str:
        logger.warning(f"ACTION query received — not yet implemented: '{query}'")
        return "I can answer questions based on indexed documents. Write/create operations are not supported yet."

    def _handle_analytical(self, query: str, user_id: str, session_id: str) -> str:
        history = self.memory_manager.get_relevant_memory(
            query=query, user_id=user_id, session_id=session_id
        )
        return self.react_agent.run(
            query=query, user_id=user_id, session_id=session_id,
            history=history,
        )

    def _handle_informational(self, query: str, user_id: str, session_id: str) -> str:
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
            query=query, chunks=chunks, history=history
        )

    def get_response(self, query: str, user_id: str, session_id: str) -> str:
        try:
            category = classify_query(query, self.llm)
            logger.info(f"Route: {category.value} | Query: '{query}'")
            return self._handlers[category](query, user_id, session_id)

        except OrchestrationError:
            raise
        except Exception as e:
            logger.exception(f"Orchestrator failed for query: '{query}'")
            raise OrchestrationError(f"Failed to process query: {e}") from e