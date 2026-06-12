import re
from utils.logger import logger
from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from prompts.react_prompts import get_react_system_prompt
from exceptions.custom_errors import ReasoningGenerationError

MAX_ITERATIONS = 5

_ACTION_RE = re.compile(r"Action\s*:\s*(\w+)", re.I)
_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.+)", re.I)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.I | re.S)


class ReActAgent:
    """
    ReAct loop — Think → Act → Observe → Think → ... → Final Answer
    Tools: retriever + calculator
    """

    def __init__(self, llm: BaseLLM, tool_agent, retriever):
        self.llm = llm
        self.tool_agent = tool_agent
        self.retriever = retriever

    def _build_history(self, history: list[tuple]) -> str:
        if not history:
            return ""
        lines = []
        for q, a in history:
            if q == "Previous conversation summary":
                lines.append(f"[CONVERSATION SUMMARY]\n{a}\n")
            else:
                lines.append(f"User: {q}\nAssistant: {a}")
        return "\n".join(lines)

    def _call_tool(self, tool_name: str, tool_input: str, user_id: str, session_id: str) -> str:
        tool_name = tool_name.strip().lower()

        if tool_name == "retriever":
            try:
                chunks = self.retriever.retrieve(
                    query=tool_input, user_id=user_id, session_id=session_id
                )
                if not chunks:
                    return "No relevant information found in documents."
                return "\n".join(c.get("text", "") for c in chunks[:3] if c.get("text"))
            except Exception as e:
                logger.warning(f"ReAct retriever failed: {e}")
                return "Retriever error — no results."

        if tool_name == "calculator":
            try:
                return self.tool_agent.execute("calculator", tool_input)
            except Exception as e:
                logger.warning(f"ReAct calculator failed: {e}")
                return f"Calculator error: {e}"

        return f"Unknown tool '{tool_name}'. Use 'retriever' or 'calculator'."

    def run(self, query: str, user_id: str, session_id: str, history: list[tuple], config: LLMConfig) -> str:
        messages = [{"role": "system", "content": get_react_system_prompt(max_iter=MAX_ITERATIONS)}]

        history_text = self._build_history(history)
        if history_text:
            messages.append({"role": "system", "content": f"CONVERSATION HISTORY:\n{history_text}"})
        messages.append({"role": "user", "content": query})

        logger.info(f"ReAct started | Query: '{query}'")

        for iteration in range(MAX_ITERATIONS):
            try:
                response = self.llm.generate(messages=messages, config=config)
            except Exception as e:
                logger.error(f"ReAct LLM call failed: {e}")
                raise ReasoningGenerationError(f"ReAct generation failed: {e}") from e

            logger.info(f"ReAct iter {iteration + 1}:\n{response}")

            final_match = _FINAL_RE.search(response)
            if final_match:
                answer = final_match.group(1).strip()
                logger.info(f"ReAct Final Answer: {answer}")
                return answer

            action_match = _ACTION_RE.search(response)
            input_match = _INPUT_RE.search(response)

            if not action_match or not input_match:
                logger.warning("ReAct: No action found, treating response as final answer.")
                return response.strip()

            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()
            logger.info(f"ReAct Action: {tool_name} | Input: '{tool_input}'")

            observation = self._call_tool(tool_name, tool_input, user_id, session_id)
            logger.info(f"ReAct Observation: {observation}")

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        logger.warning("ReAct: Max iterations reached.")
        try:
            final_response = self.llm.generate(messages=messages + [
                {"role": "user", "content": "Please provide your Final Answer now based on what you have found."}
            ], config=config)
            final_match = _FINAL_RE.search(final_response)
            if final_match:
                return final_match.group(1).strip()
            return final_response.strip()
        except Exception:
            return "I could not complete the reasoning chain. Please try rephrasing."