from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    ):
        try:
            self.model = CrossEncoder(
                model_name
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to load reranker model"
            ) from e


    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        if not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if not chunks:

            raise ValueError(
                "chunks cannot be empty"
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be positive"
            )

        try:
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

        except Exception as e:

            raise RuntimeError(
                "Failed to rerank chunks"
            ) from e