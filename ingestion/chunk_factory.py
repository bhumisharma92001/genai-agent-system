import uuid
from exceptions.custom_errors import InvalidInputError, ChunkingError


class ChunkFactory:

    def create_text_chunk(self, text: str, metadata: dict) -> dict:
        if not text or not text.strip():
            raise InvalidInputError("Chunk text cannot be empty")
        try:
            chunk_key = (
                f"{metadata.get('user_id', '')}:"
                f"{metadata.get('session_id', '')}:"
                f"{metadata.get('document_id', '')}:"
                f"{metadata.get('source', '')}:"
                f"{metadata.get('chunk_index', '')}:"
                f"{text[:200]}"
            )
            return {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_key)),
                "chunk_type": "text",
                "text": text,
                "metadata": {**metadata, "chunk_type": "text"},
            }
        except InvalidInputError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to create text chunk: {e}") from e

    def create_table_chunk(self, summary: str, metadata: dict) -> dict:
        if not summary or not summary.strip():
            raise InvalidInputError("Table summary cannot be empty")
        try:
            chunk_key = (
                f"{metadata.get('user_id', '')}:"
                f"{metadata.get('session_id', '')}:"
                f"{metadata.get('document_id', '')}:"
                f"{metadata.get('source', '')}:"
                f"{metadata.get('chunk_index', '')}:"
                f"{summary[:200]}"
            )
            return {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_key)),
                "chunk_type": "table",
                "text": summary,
                "metadata": {**metadata, "chunk_type": "table"},
            }
        except InvalidInputError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to create table chunk: {e}") from e

    def create_table_row_chunk(self, text: str, metadata: dict) -> dict:
        if not text or not text.strip():
            raise InvalidInputError("Table row text cannot be empty")
        try:
            table_title = metadata.get("table_title", "")
            headers = metadata.get("column_headers", [])

            enriched_parts = []
            if table_title:
                enriched_parts.append(f"Table: {table_title}")
            if headers:
                enriched_parts.append(f"Columns: {' | '.join(str(h) for h in headers)}")
            enriched_parts.append(f"Row: {text}")

            enriched_text = "\n".join(enriched_parts)

            chunk_key = (
                f"{metadata.get('user_id', '')}:"
                f"{metadata.get('session_id', '')}:"
                f"{metadata.get('document_id', '')}:"
                f"{metadata.get('source', '')}:"
                f"{metadata.get('chunk_index', '')}:"
                f"{enriched_text[:200]}"
            )
            return {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_key)),
                "chunk_type": "table_row",
                "text": enriched_text,
                "metadata": {**metadata, "chunk_type": "table_row"},
            }
        except InvalidInputError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to create table row chunk: {e}") from e