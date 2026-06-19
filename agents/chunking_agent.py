import asyncio
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger
from exceptions.custom_errors import ChunkingError, GenAIError


class ChunkingAgent:

    def __init__(self, architect_agent):
        self.architect_agent = architect_agent

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)

    def chunk_table(self, table_text: str) -> list[str]:
        return [table_text]

    async def chunk(self, text: str, tables: list[str], file_path: str, page_no: int = None) -> dict:
        try:
            chunks = []
            identify_config = {"chunk_size": None, "overlap": None, "reasoning": "table/empty text — LLM bypassed"}
            if text and text.strip():
                config = await self.architect_agent.decide(page_text=text, file_name=Path(file_path).name, page_no=page_no)
                identify_config = config
                chunks.extend(self.chunk_text(text=text, chunk_size=config.get("chunk_size", 800), overlap=config.get("overlap", 150)))
            for table in tables:
                chunks.extend(self.chunk_table(table))
            return {"chunks": chunks, "config": identify_config}
        except GenAIError:
            raise
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            raise ChunkingError(f"Failed to chunk: {e}") from e