import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger import logger


class ArchitectAgent:

    def __init__(self, llm):
        self.llm = llm

    def decide(self, preview: str, file_name: str) -> dict:
        try:
            resp = self.llm.invoke([
                SystemMessage(content="""You are a document analysis expert.
                Analyze the document and output ONLY a JSON config. No explanation.

                Rules:
                    - Financial tables found → atomic_tables: true
                    - Narrative/prose → threshold: 0.75
                    - Legal/technical → threshold: 0.85
                    - FAQ/structured → threshold: 0.65

                Output ONLY this JSON:
                    {
                        "chunking_strategy": "semantic",
                        "threshold": 0.75,
                        "atomic_tables": true
                    }"""),
                HumanMessage(content=f"""File: {file_name} 
                             Preview: {preview[:500]}""")])

            config = json.loads(resp.content.strip())
            logger.info(f"ArchitectAgent config: {config}")
            return config

        except Exception as e:
            logger.warning(f"ArchitectAgent failed: {e}. Using defaults.")
            return {
                "chunking_strategy": "semantic",
                "threshold": 0.75,
                "atomic_tables": True
            }