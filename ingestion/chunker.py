from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class TextChunker:

    def __init__(
        self,
        chunk_size: int=700,
        overlap: int=100
    ):
 
        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be positive"
            )

        if overlap < 0:

            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= chunk_size:

            raise ValueError(
                "overlap must be smaller "
                "than chunk_size"
            )

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

    def split_text(
        self,
        text: str
    ) -> list[str]:

        if not text or not text.strip():

            raise ValueError(
                "Input text cannot be empty"
            )

        try:

            return self.text_splitter.split_text(
                text
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to split text"
            ) from e