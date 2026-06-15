from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from prompts.react_prompts import get_react_system_prompt
from utils.conversation import build_history
from utils.logger import logger
from exceptions.custom_errors import ReasoningGenerationError


class ReActAgent:

    def __init__(self, llm, retriever, registry):
        self.llm = llm
        self.retriever = retriever
        self.registry = registry
        self._graph_cache: dict = {}

    def _make_tools(self, user_id: str, session_id: str) -> list:
        retriever = self.retriever  
        registry = self.registry

        @tool
        def retriever_tool(query: str) -> str:
            """Search indexed documents using short keyword queries of 2-5 words."""
            chunks = retriever.retrieve(query=query, user_id=user_id, session_id=session_id)
            if not chunks:
                return "No relevant information found in documents."
            return "\n".join(c.get("text", "") for c in chunks if c.get("text"))

        @tool
        def calculator_tool(expression: str) -> str:
            """Evaluate arithmetic expressions. Numbers and operators only e.g. (56 + 87) / 2"""
            return registry.get("calculator")(expression)

        return [retriever_tool, calculator_tool]

    def run(self, query: str, user_id: str, session_id: str, history: list[tuple], config=None) -> str:
        try:
            cache_key = (user_id, session_id)
            if cache_key not in self._graph_cache:
                tools = self._make_tools(user_id, session_id)
                self._graph_cache[cache_key] = create_agent(
                    model=self.llm,
                    tools=tools,
                    system_prompt=get_react_system_prompt(),
                )
            graph = self._graph_cache[cache_key]

            history_text = build_history(history)
            messages = []
            if history_text:
                messages.append(SystemMessage(content=f"CONVERSATION HISTORY:\n{history_text}"))
            messages.append(HumanMessage(content=query))

            result = graph.invoke({"messages": messages})  
            return result["messages"][-1].content           
        except Exception as e:
            logger.error(f"ReAct failed: {e}")
            raise ReasoningGenerationError(f"ReAct failed: {e}") from e