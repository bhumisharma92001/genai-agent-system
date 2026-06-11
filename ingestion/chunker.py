import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger
from exceptions.custom_errors import ChunkingError, ConfigurationError

class TextChunker:
    def __init__(self):
        try:
            self.chunk_size = int(os.getenv("CHUNK_SIZE", 700))
            self.overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        except ValueError:
            raise ConfigurationError("CHUNK_SIZE and CHUNK_OVERLAP in .env must be integers.")

        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be > 0.")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ConfigurationError("Overlap must be 0 <= overlap < chunk_size.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"TextChunker configured: Size={self.chunk_size}, Overlap={self.overlap}")

    def split_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            raise ChunkingError(f"Splitting failed: {e}") from e