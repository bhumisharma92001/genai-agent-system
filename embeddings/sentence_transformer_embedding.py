from sentence_transformers import SentenceTransformer
from embeddings.base import BaseEmbedding

class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {model_name}") from e

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        try:
            if is_query:
                text = "Represent this sentence for searching relevant passages: " + text
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError("Failed to generate embedding") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts cannot be empty")
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError("Failed to generate batch embeddings") from e