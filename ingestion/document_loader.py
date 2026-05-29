from pathlib import Path
from ingestion.text_extractor import TextExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.table_summarizer import TableSummarizer


class DocumentLoader:

    def __init__(
        self,
        chunker: TextChunker
    ):

        self.text_extractor = (
            TextExtractor()
        )

        self.table_extractor = (
            TableExtractor()
        )

        self.chunker = chunker

        self.chunk_factory = (
            ChunkFactory()
        )

        self.table_summarizer = (
            TableSummarizer()
        )

    def load(
        self,
        file_path: str
    ) -> list[dict]:

        if not file_path:

            raise ValueError(
                "file_path cannot be empty"
            )

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        metadata = {
            "document_id": path.name,
            "source": str(file_path)
        }

        chunks = []

        # --------------------------------
        # TEXT PROCESSING
        # --------------------------------

        try:

            text = self.text_extractor.extract(
                file_path
            )

            text_chunks = (
                self.chunker.split_text(
                    text=text
                )
            )

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

            raise RuntimeError(
                "Failed to process text"
            ) from e

        # --------------------------------
        # TABLE PROCESSING
        # --------------------------------

        try:

            extracted_tables = (
                self.table_extractor.extract(
                    file_path
                )
            )

            for table in extracted_tables:

                summary = (
                    self.table_summarizer
                    .summarize(
                        table["dataframe"]
                    )
                )

                table_chunk = (
                    self.chunk_factory
                    .create_table_chunk(
                        summary=summary,
                        metadata={
                            **metadata,
                            "page": table["page"]
                        }
                    )
                )

                chunks.append(table_chunk)

        except Exception as e:

            raise RuntimeError(
                "Failed to process tables"
            ) from e

        return chunks