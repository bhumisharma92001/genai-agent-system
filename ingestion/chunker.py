import uuid

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class Chunker:

    def __init__(
        self,
        chunk_size: int = 700,
        overlap: int = 150
    ):

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

    def chunk_text(
        self,
        text: str,
        metadata: dict
    ) -> list[dict]:

        chunks = (
            self.text_splitter.split_text(text)
        )

        formatted_chunks = []

        for chunk in chunks:

            formatted_chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_type": "text",
                    "text": chunk,
                    "metadata": metadata
                }
            )

        return formatted_chunks

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