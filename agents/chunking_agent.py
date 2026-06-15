import os
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from utils.logger import logger
from exceptions.custom_errors import ChunkingError


class EmbeddingAdapter(Embeddings):
    """
    Adapts SentenceTransformerEmbedding to the LangChain Embeddings interface
    required by SemanticChunker.
    """
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed(text, is_query=True)


class ChunkingAgent:

    def __init__(self, embedding_model, architect_agent):
        self.architect_agent = architect_agent
        self.adapted_embeddings = EmbeddingAdapter(embedding_model)

    def chunk_text(self, text: str, threshold: float) -> list[str]:
        splitter = SemanticChunker(
            embeddings=self.adapted_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold * 100
        )
        return splitter.split_text(text)

    def chunk_table(self, table_text: str) -> list[str]:
        return [table_text]

    def chunk(self, text: str, tables: list[str], file_path: str) -> list[str]:
        preview_source = text if text.strip() else (tables[0] if tables else "")
        preview_size = min(len(preview_source) // 10, 1000)
        config = self.architect_agent.decide(
            preview=preview_source[:preview_size],
            file_name=Path(file_path).name
        )
        chunks = []
        if text and text.strip():
            text_chunks = self.chunk_text(
                text=text,
                threshold=config.get("threshold", 0.75)
            )
            chunks.extend(text_chunks)
        if config.get("atomic_tables", True):
            for table in tables:
                chunks.extend(self.chunk_table(table))
        return chunks