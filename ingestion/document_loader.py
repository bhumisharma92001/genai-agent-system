from pathlib import Path
from ingestion.text_extractor import TextExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.table_summarizer import TableSummarizer
from utils.logger import logger

class DocumentLoader:

    def __init__(
        self,
        chunker: TextChunker,
        text_extractor: TextExtractor,
        table_extractor: TableExtractor,
        chunk_factory: ChunkFactory,
        table_summarizer: TableSummarizer,
    ):
        self.chunker = chunker
        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.chunk_factory = chunk_factory
        self.table_summarizer = table_summarizer

    def load(self,file_path: str) -> list[dict]:
        
        if not file_path:
            raise ValueError("file_path cannot be empty")
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = {"document_id": path.name,"source": str(file_path)}
        chunks = []
        logger.info(f"Loading document: {file_path}")

        try:
            text = self.text_extractor.extract(file_path)
            if text.strip():
                text_chunks = (self.chunker.split_text(text=text))

                for chunk_text in text_chunks:

                    chunk = (
                        self.chunk_factory
                        .create_text_chunk(
                            text=chunk_text,
                            metadata=metadata
                        )
                    )

                    chunks.append(chunk)

        except Exception as e:
            raise RuntimeError("Failed to process text") from e

        logger.info("Extracting tables")

        try:
            extracted_tables = (self.table_extractor.extract(file_path))
            for table in extracted_tables:
                summary = (self.table_summarizer.summarize(table["dataframe"]))

                table_chunk = (
                    self.chunk_factory
                    .create_table_chunk(
                        summary=summary,
                        metadata={
                            **metadata,
                            "page": table["page"],
                        }
                    )
                )

                chunks.append(table_chunk)
                dataframe = table["dataframe"]

                for row_index, row in dataframe.iterrows():
                    row_text = (self.table_summarizer.summarize_row(row))
                    row_chunk = (
                        self.chunk_factory
                        .create_table_row_chunk(
                            text=row_text,
                            metadata={**metadata,"page": table["page"],"row_index": int(row_index)}
                        )
                    )
                    chunks.append(row_chunk)

        except Exception as e:

            raise RuntimeError("Failed to process tables") from e
        logger.info(f"Created {len(chunks)} chunks")
        return chunks