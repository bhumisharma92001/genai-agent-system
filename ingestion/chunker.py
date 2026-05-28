import uuid

class Chunker:

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        metadata: dict
    ) -> list[dict]:

        words = text.split()

        if not words:
            return []

        chunks = []

        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk_words = words[start:end]

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_type": "text",
                    "text": " ".join(chunk_words),
                    "metadata": metadata
                }
            )

            if end >= len(words):
                break

            start = end - self.overlap

        return chunks

    @staticmethod
    def chunk_table_summary(
        summary: str,
        metadata: dict
    ) -> dict:

        return {
            "chunk_id": str(uuid.uuid4()),
            "chunk_type": "table_summary",
            "text": summary,
            "metadata": metadata
        }