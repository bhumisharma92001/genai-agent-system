from sentence_transformers import SentenceTransformer
from embeddings.base import BaseEmbedding
from utils.logger import logger
from exceptions.custom_errors import EmbeddingError, ModelLoadError, InvalidInputError

class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(self, model_name: str):
        if not model_name or not model_name.strip():
            raise InvalidInputError("model_name cannot be empty")
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {model_name}") from e

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        if not text or not text.strip():
            raise InvalidInputError("Input text cannot be empty")
        try:
            if is_query:
                text = "Represent this sentence for searching relevant passages: " + text
            return self.model.encode(text, normalize_embeddings=True).tolist()
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError("Failed to generate embedding") from e

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            raise InvalidInputError("texts cannot be empty")
        try:
            if is_query:
                texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
            return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise EmbeddingError("Failed to generate batch embeddings") from e