from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self, model_name: str):
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        try:
            self.model = CrossEncoder(model_name)

        except Exception as e:
            raise RuntimeError("Failed to load reranker model") from e

    def rerank(self,query: str,chunks: list[dict],top_k: int = 3) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            pairs = [[query, chunk["text"]] for chunk in chunks]
            scores = self.model.predict(pairs)
            if not isinstance(scores, list) and hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif isinstance(scores, float) or isinstance(scores, int):
                scores = [scores]
            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1],reverse=True)

            reranked_chunks = []
            for chunk, score in scored_chunks[:top_k]:
                chunk["rerank_score"] = float(score)
                reranked_chunks.append(chunk)
            return reranked_chunks

        except Exception as e:
            raise RuntimeError("Failed to rerank chunks") from e