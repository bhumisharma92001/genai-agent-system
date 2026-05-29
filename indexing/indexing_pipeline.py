from ingestion.document_loader import (
    DocumentLoader
)

from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from vector_db.chroma_db import ChromaDB


class IndexingPipeline:

    def __init__(
        self,
        loader: DocumentLoader,
        embedding_model: SentenceTransformerEmbedding,
        vector_db: ChromaDB
    ):

        self.loader = loader

        self.embedding_model = (
            embedding_model
        )

        self.vector_db = vector_db

    def index_document(
        self,
        file_path: str
    ) -> None:

        chunks = self.loader.load(
            file_path
        )

        ids = []

        documents = []

        embeddings = []

        metadatas = []

        for chunk in chunks:

            embedding = (
                self.embedding_model.embed(
                    chunk["text"]
                )
            )

            ids.append(
                chunk["chunk_id"]
            )

            documents.append(
                chunk["text"]
            )

            embeddings.append(
                embedding
            )

            metadatas.append(
                chunk["metadata"]
            )

        self.vector_db.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )