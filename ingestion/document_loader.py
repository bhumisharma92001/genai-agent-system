from pathlib import Path
from ingestion.text_extractor import TextExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunker import TextChunker
from ingestion.chunk_factory import ChunkFactory
from ingestion.table_summarizer import TableSummarizer
from ingestion.table_storage import TableStorage


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

        self.table_storage = (
            TableStorage()
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
            if file_path.endswith(
                (".pdf", ".docx")
            ):
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

                table_name = (
                    self.table_storage.store(
                    table["dataframe"]
                    )
                )

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
                            "page": table["page"],
                            "table_name": table_name
                        }
                    )
                )

                chunks.append(table_chunk)

                dataframe = table["dataframe"]

                for row_index, row in dataframe.iterrows():

                    row_text = (
                        self.table_summarizer
                        .summarize_row(row)
                    )

                    row_chunk = (
                        self.chunk_factory
                        .create_table_row_chunk(
                            text=row_text,
                            metadata={
                                **metadata,
                                "page": table["page"],
                                "table_name": table_name,
                                "row_index": int(row_index)
                            }
                        )
                    )

                    chunks.append(
                        row_chunk
                    )

        except Exception as e:

            raise RuntimeError(
                "Failed to process tables"
            ) from e

        return chunks