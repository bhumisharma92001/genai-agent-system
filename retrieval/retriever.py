from retrieval.reranker import Reranker

class Retriever:

    def __init__( self, embedding_model, vector_db, reranker: Reranker ):
        
        self.embedding_model = (
            embedding_model
        )

        self.vector_db = (
            vector_db
        )

        self.reranker = (
            reranker
        )
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

        if not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:

            raise ValueError(
                "k must be positive"
            )

        try:

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

            reranked_chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=10
            )

            return reranked_chunks

        except Exception as e:

            raise RuntimeError(
                "Failed to retrieve chunks"
            ) from e