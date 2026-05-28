# ingestion/document_loader.py

from pathlib import Path

from ingestion.text_extractor import TextExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunker import Chunker


class DocumentLoader:

    def __init__(self):

        self.text_extractor = TextExtractor()

        self.table_extractor = TableExtractor()

        self.chunker = Chunker()

    def load(
        self,
        file_path: str
    ) -> list[dict]:

        path = Path(file_path)

        metadata = {
            "document_id": path.name,
            "source": str(file_path)
        }

        chunks = []

        # --------------------------------
        # TEXT EXTRACTION
        # --------------------------------

        try:

            text = self.text_extractor.extract(
                file_path
            )

            text_chunks = self.chunker.chunk_text(
                text=text,
                metadata=metadata
            )

            chunks.extend(text_chunks)

        except Exception:

            pass

        # --------------------------------
        # TABLE EXTRACTION
        # --------------------------------

        table_chunks = self.table_extractor.extract(
            file_path
        )

        for table_chunk in table_chunks:

            chunk = self.chunker.chunk_table_summary(
                summary=table_chunk["text"],
                metadata={
                    **metadata,
                    **table_chunk["metadata"]
                }
            )

            chunks.append(chunk)

        return chunks