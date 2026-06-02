from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig

class SummarizationAgent:

    def __init__(self,llm: BaseLLM):
        self.llm = llm
        self.config = LLMConfig(temperature=0.2,top_p=0.8,max_tokens=256)

    def summarize(self,conversation: str) -> str:
        prompt = f"""
            Summarize the following conversation.
            Conversation:{conversation}
        """
        messages = [{"role": "user","content": prompt}]
        return self.llm.generate(
            messages=messages,
            config=self.config
        )
