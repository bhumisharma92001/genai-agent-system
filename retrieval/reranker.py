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
            ]#cross encoder needs pair of query and text for input

            scores = self.model.predict(
                pairs
            )

            scored_chunks = list(
                zip(chunks, scores)
            )#same index is combined

            scored_chunks.sort(
                key=lambda x: x[1],#x is chunk and x[1]is score
                reverse=True
            )

            reranked_chunks = []

            for chunk, score in scored_chunks[:top_k]:
                print("RAW SCORE:", score)

                chunk["rerank_score"] = float(score)

                reranked_chunks.append(
                chunk
            )
            return reranked_chunks

        except Exception as e:

            raise RuntimeError(
                "Failed to rerank chunks"
            ) from e