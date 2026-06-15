from pathlib import Path
from ingestion.text_extractor import TextExtractor
from ingestion.table_extractor import TableExtractor
from ingestion.chunk_factory import ChunkFactory
from utils.logger import logger
from exceptions.custom_errors import DocumentLoadError, InvalidInputError
from agents.chunking_agent import ChunkingAgent


class DocumentLoader:

    def __init__(
        self,
        text_extractor: TextExtractor,
        table_extractor: TableExtractor,
        chunk_factory: ChunkFactory,
        chunking_agent: ChunkingAgent,
    ):

        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.chunk_factory = chunk_factory
        self.chunking_agent = chunking_agent

    def load(
        self,
        file_path: str,
        user_id: str,
        session_id: str,
        document_id: str = None,
    ) -> list[dict]:
        if not file_path:
            raise InvalidInputError("file_path cannot be empty")
        path = Path(file_path)
        if not path.exists():
            raise InvalidInputError(f"File not found: {file_path}")

        doc_id = document_id or path.name
        base_metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "document_id": doc_id,
            "source": str(file_path),
        }
        chunks = []
        logger.info(f"Loading document: {file_path}")

        file_extension = path.suffix.lower()
        is_spreadsheet = file_extension in (".csv", ".xlsx", ".xls")

        if not is_spreadsheet:
            logger.info("Extracting raw text for non-spreadsheet file")
            try:
                text = self.text_extractor.extract(file_path)
                if text.strip():
                    text_chunks = self.chunking_agent.chunk(
                        text=text,
                        tables=[],
                        file_path=file_path
                    )
                    for i, chunk_text in enumerate(text_chunks):
                        chunks.append(
                            self.chunk_factory.create_text_chunk(
                                text=chunk_text,
                                metadata={**base_metadata, "chunk_index": i},
                            )
                        )
            except InvalidInputError:
                raise
            except Exception as e:
                logger.error(f"Text processing failed for: {file_path} — {e}")
                raise DocumentLoadError(f"Failed to process text: {e}") from e
        else:
            logger.info(
                f"Skipping text extraction for structured spreadsheet: {file_extension}"
            )

        logger.info("Extracting tables/structured data")
        try:
            extracted_tables = self.table_extractor.extract(file_path)
            
            for table in extracted_tables:
                dataframe = table["dataframe"]
                table_title = table.get("table_title", "")
                column_headers = table.get("column_headers", list(dataframe.columns))
                page = table.get("page", 1)
                table_metadata = {
                    **base_metadata,
                    "page": page,
                    "table_title": table_title,
                    "column_headers": column_headers,
                }
                table_text = dataframe.to_string(index=False)
                table_chunks = self.chunking_agent.chunk(
                    text="",
                    tables=[table_text],
                    file_path=file_path
                )
                for i, table_chunk in enumerate(table_chunks):
                    chunks.append(
                        self.chunk_factory.create_table_chunk(
                            summary=table_chunk,
                            metadata={**table_metadata, "chunk_index": len(chunks)},
                        )
                    )

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Table processing failed for: {file_path} — {e}")
            raise DocumentLoadError(f"Failed to process tables: {e}") from e

        logger.info(f"Created {len(chunks)} chunks")
        return chunks