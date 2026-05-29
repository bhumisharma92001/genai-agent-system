from sentence_transformers import SentenceTransformer

from embeddings.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        self.model_name = model_name

        try:

            self.model = SentenceTransformer(
                model_name
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model: "
                f"{model_name}"
            ) from e

    def embed(
        self,
        text: str
    ) -> list[float]:
        if not text or not text.strip():

            raise ValueError(
                "Input text cannot be empty"
            )

        try:

            embedding = self.model.encode(
                text
            )

            return embedding.tolist()

        except Exception as e:

            raise RuntimeError(
                "Failed to generate embedding"
            ) from e