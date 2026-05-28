from sentence_transformers import (
    CrossEncoder
)


class Reranker:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    ):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        pairs = [
            [query, chunk["text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(
            pairs
        )

        scored_chunks = list(
            zip(chunks, scores)
        )

        scored_chunks.sort(
            key=lambda x: x[1],
            reverse=True
        )

        reranked_chunks = [
            chunk
            for chunk, score in scored_chunks[:top_k]
        ]

        return reranked_chunks