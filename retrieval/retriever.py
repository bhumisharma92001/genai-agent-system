from embeddings.base import BaseEmbedding
from retrieval.reranker import Reranker
from utils.logger import logger
from vector_db.base_vector_db import BaseVectorDB
class Retriever:

    def __init__( 
        
        self,
        embedding_model : BaseEmbedding,
        vector_db : BaseVectorDB,
        reranker: Reranker,
        retrieval_k: int = 20,
        rerank_k: int = 5
    ):
        if rerank_k > retrieval_k:
            raise ValueError(
                "rerank_k cannot be greater than retrieval_k"
            )
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        
    def retrieve(self,query: str,) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        try:
            query_embedding = (self.embedding_model.embed(query))
            results = self.vector_db.query(embedding=query_embedding,k=self.retrieval_k)
            chunks = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            for document, metadata in zip(documents,metadatas):
                chunks.append({"text": document,"metadata": metadata})
            reranked_chunks = self.reranker.rerank(query=query,chunks=chunks,top_k=self.rerank_k)
            return reranked_chunks

        except Exception as e:
            logger.exception("Failed to retrieve chunks") 

            raise RuntimeError("Failed to retrieve chunks" )from e