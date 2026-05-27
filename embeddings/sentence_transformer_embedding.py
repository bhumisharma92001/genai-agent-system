from sentence_transformers import SentenceTransformer

from embeddings.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed(
        self,
        text: str
    ) -> list[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()