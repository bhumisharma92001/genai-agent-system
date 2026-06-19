import uuid
from exceptions.custom_errors import InvalidInputError, ChunkingError


class ChunkFactory:

    def _build_chunk_key(self, text: str, metadata: dict) -> str:
        return (
            f"{metadata.get('user_id', '')}:"
            f"{metadata.get('session_id', '')}:"
            f"{metadata.get('document_id', '')}:"
            f"{metadata.get('source', '')}:"
            f"{metadata.get('chunk_index', '')}:"
            f"{metadata.get('page_number', '')}:"
            f"{text[:200]}"
        )

    def create_text_chunk(self, text: str, metadata: dict) -> dict:
        if not text or not text.strip():
            raise InvalidInputError("Chunk text cannot be empty")
        try:
            return {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, self._build_chunk_key(text, metadata))),
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
            return {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, self._build_chunk_key(summary, metadata))),
                "chunk_type": "table",
                "text": summary,
                "metadata": {**metadata, "chunk_type": "table"},
            }
        except InvalidInputError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to create table chunk: {e}") from e
