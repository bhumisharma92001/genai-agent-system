from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from vector_db.chroma_db import ChromaDB

from retrieval.reranker import Reranker


class Retriever:

    def __init__(self):

        self.embedding_model = (
            SentenceTransformerEmbedding()
        )

        self.vector_db = ChromaDB()

        self.reranker = Reranker()

    def keyword_score(
        self,
        query: str,
        text: str
    ) -> int:

        query_words = set(
            query.lower().split()
        )

        text_words = set(
            text.lower().split()
        )

        return len(
            query_words.intersection(
                text_words
            )
        )

    def retrieve(
        self,
        query: str,
        k: int = 10
    ) -> list[dict]:

        query_embedding = (
            self.embedding_model.embed(query)
        )

        results = self.vector_db.query(
            embedding=query_embedding,
            k=k
        )

        chunks = []

        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            semantic_score = 1 - distance

            keyword_boost = (
                self.keyword_score(
                    query=query,
                    text=document
                )
            )

            final_score = (
                semantic_score
                +
                (0.1 * keyword_boost)
            )

            chunks.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "score": final_score
                }
            )

        chunks.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        reranked_chunks = (
            self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=3
            )
        )

        return reranked_chunks