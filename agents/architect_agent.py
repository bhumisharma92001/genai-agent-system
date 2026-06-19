import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger import logger
from prompts.architect_agent_prompts import get_architect_system_prompt


class ArchitectAgent:

    def __init__(self, llm):
        self.llm = llm

    def _messages(self, page_text: str, file_name: str) -> list:
        return [
            SystemMessage(content=get_architect_system_prompt()),
            HumanMessage(content=f"""File: {file_name}
            Page text: {page_text}"""),
        ]

    async def decide(self, page_text: str, file_name: str, page_no: int = None) -> dict:
        try:
            resp = await self.llm.ainvoke(self._messages(page_text, file_name))

            config = json.loads(resp.content.strip())
            logger.info(f"ArchitectAgent config for {file_name} page {page_no}: {config}")
            return config

        except Exception as e:
            logger.warning(f"ArchitectAgent failed: {e}. Using defaults.")
            return {"chunk_size": 800, "overlap": 150}