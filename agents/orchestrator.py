from langgraph.graph import StateGraph, END
from typing import TypedDict
from llm.llm_config import LLMConfig
from utils.logger import logger


class State(TypedDict):
    query: str
    user_id: str
    session_id: str

    route: str
    tool: str | None

    chunks: list
    answer: str


class Orchestrator:

    def __init__(self, router, retriever, tool_agent, reasoning_agent, memory_manager):
        self.router = router
        self.retriever = retriever
        self.tool_agent = tool_agent
        self.reasoning_agent = reasoning_agent
        self.memory_manager = memory_manager

        self.graph = self._build()

    # -------------------------
    # GRAPH
    # -------------------------
    def _build(self):

        g = StateGraph(State)

        g.add_node("route", self._route)
        g.add_node("tool", self._tool)
        g.add_node("rag", self._rag)
        g.add_node("reason", self._reason)
        g.add_node("final", self._final)

        g.set_entry_point("route")

        g.add_conditional_edges(
            "route",
            self._decide,
            {
                "tool": "tool",
                "rag": "rag",
                "reasoning": "reason"
            }
        )

        g.add_edge("tool", "final")
        g.add_edge("rag", "reason")
        g.add_edge("reason", "final")
        g.add_edge("final", END)

        return g.compile()

    # -------------------------
    # NODES
    # -------------------------

    def _route(self, state: State):
        try:
            result = self.router.classify(
                state["query"],
                config=LLMConfig(temperature=0, max_tokens=16)
            )
            route = str(result.get("route", "rag")).strip().lower()
            state["route"] = route
            state["tool"] = result.get("tool")
        except Exception:
            state["route"] = "rag"
            state["tool"] = None
        logger.info(f"[ROUTE NODE] query={state['query']!r}")
        logger.info(
            f"QueryRouter selected route={state['route']} tool={state['tool']} query={state['query']!r}"
        )
        return state

    def _decide(self, state: State):
        if state["route"] not in {"tool", "rag", "reasoning"}:
            return "rag"
        return state["route"]

    def _tool(self, state: State):
        try:
            result = self.tool_agent.execute(state["query"])
            if isinstance(result, dict) and "result" in result:
                state["answer"] = str(result["result"])
                logger.info(f"Tool execution completed query={state['query']!r} answer={state['answer']}")
            else:
                state["answer"] = str(result)
        except Exception as exc:
            state["answer"] = f"Error: {str(exc)}"
        logger.info(f"[TOOL NODE] executing calculator for query={state['query']!r}")
        return state

    def _rag(self, state: State):
        state["chunks"] = self.retriever.retrieve(
            query=state["query"],
            user_id=state["user_id"],
            session_id=state["session_id"]
        )
        if not state["chunks"]:
            state["answer"] = "No relevant context found."
            return state
        logger.info(f"[RAG NODE] retrieving chunks for query={state['query']!r}")
        logger.info(f"[RAG NODE] chunks_found={len(state['chunks'])}")
        return state

    def _reason(self, state: State):
        try:
            history = self.memory_manager.get_recent_interactions(
                state["user_id"],
                state["session_id"]
            )
            summaries = self.memory_manager.get_summaries(
                user_id=state["user_id"],
                session_id=state["session_id"]
            )

            state["answer"] = self.reasoning_agent.answer(
                query=state["query"],
                chunks=state.get("chunks", []),
                conversation_history=history,
                summaries=summaries,
                config=LLMConfig()
            )
        except Exception as exc:
            state["answer"] = f"Error: {str(exc)}"
        logger.info(f"[REASON NODE] reasoning started for query={state['query']!r}")
        return state

    def _final(self, state: State):
        if state["route"] == "tool":
            logger.info("Skipping memory save for tool route")
            return state

        self.memory_manager.save_interaction(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["query"],
            answer=state["answer"]
        )

        self.memory_manager.summarize_in_background(
            user_id=state["user_id"],
            session_id=state["session_id"]
        )
        logger.info(f"[FINAL NODE] route={state['route']} answer={state['answer']}")
        return state

    # -------------------------
    # RUN
    # -------------------------

    def run(self, query, user_id, session_id):

        state = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "route": "",
            "tool": None,
            "chunks": [],
            "answer": ""
        }
        logger.info(f"\n\n🔥 NEW REQUEST: {query}\n")
        result = self.graph.invoke(state)
        return result["answer"]