from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
from tools.tool_registry import ToolRegistry

class ToolAgent:

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.registry = ToolRegistry()

    def route(self, query: str) -> str:
        prompt = f"""
        You are a query routing agent.
        Available routes:
        calculator
            - arithmetic calculations
            - add, subtract, multiply, divide
            - calculations that can be solved WITHOUT document data

        rag
            - document questions
            - PDF questions
            - table questions
            - spreadsheet questions
            - report questions
            - any question requiring document retrieval

        Examples:

            Query: 2+2
            Route: calculator

            Query: add 4 and 7
            Route: calculator

            Query: what is 100 divided by 5
            Route: calculator

            Query: what is the revenue in Q4
            Route: rag

            Query: what is the average salary in the employee table
            Route: rag

            Query: summarize the report
            Route: rag

            Return ONLY:
                calculator or rag

            Query:{query}"""

        response = self.llm.generate(messages=[{"role": "user","content": prompt}],
            config=LLMConfig(temperature=0,max_tokens=5))
        return response.strip().lower()

    def execute(self, query: str):
        calculator = self.registry.get_tool("calculator")
        return calculator.execute({"query": query})