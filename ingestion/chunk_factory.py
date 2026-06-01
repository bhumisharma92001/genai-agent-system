import uuid

class ChunkFactory:

    def create_text_chunk(
        self,
        text: str,
        metadata: dict
    ) -> dict:

        if not text or not text.strip():

            raise ValueError(
                "Chunk text cannot be empty"
            )

        return {
            "chunk_id": str(uuid.uuid4()),
            "chunk_type": "text",
            "text": text,
            "metadata": metadata
        }

    def create_table_chunk(
        self,
        summary: str,
        metadata: dict
    ) -> dict:

        if not summary or not summary.strip():

            raise ValueError(
                "Table summary cannot be empty"
            )

        return {
            "chunk_id": str(uuid.uuid4()),
            "chunk_type": "table",
            "text": summary,
            "metadata": metadata
        }

    def create_table_row_chunk(
        self,
        text: str,
        metadata: dict
    ) -> dict:

        if not text or not text.strip():

            raise ValueError(
                "Table row text cannot be empty"
            )

        return {
            "chunk_id": str(uuid.uuid4()),
            "chunk_type": "table_row",
            "text": text,
            "metadata": metadata
        }