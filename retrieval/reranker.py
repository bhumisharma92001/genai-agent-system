import numpy as np
from sentence_transformers import CrossEncoder
from utils.logger import logger
from exceptions.custom_errors import RerankError, InvalidInputError, ModelLoadError

class Reranker:
    def __init__(self, model_name: str):
        if not model_name or not model_name.strip():
            raise InvalidInputError("model_name cannot be empty")
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Reranker model successfully initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            raise ModelLoadError(f"Failed to load reranker model: {e}") from e

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        score_threshold: float | None = None
    ) -> list[dict]:
        if not query or not query.strip():
            raise InvalidInputError("query cannot be empty")
        if not chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            pairs = [
                [query, chunk.get("text") or chunk.get("page_content") or str(chunk)]
                for chunk in chunks
            ]
            scores = self.model.predict(pairs)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif isinstance(scores, (float, int, np.float32, np.float64)):
                scores = [float(scores)]
            else:
                scores = list(scores)

            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            reranked_chunks = []
            for chunk, score in scored_chunks:
                if len(reranked_chunks) >= top_k:
                    break
                if score_threshold is not None and float(score) < score_threshold:
                    continue  
                cloned_chunk = chunk.copy() 
                cloned_chunk["rerank_score"] = float(score)
                reranked_chunks.append(cloned_chunk)

            logger.info(f"Reranking complete. Passed {len(reranked_chunks)}/{len(chunks)} chunks.")
            return reranked_chunks

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Rerank engine fault: {str(e)}")
            raise RerankError(f"Failed to rerank chunks: {e}") from e