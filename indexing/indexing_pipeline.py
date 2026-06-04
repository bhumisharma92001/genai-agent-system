from ingestion.document_loader import DocumentLoader
from embeddings.base import BaseEmbedding
from vector_db.base_vector_db import BaseVectorDB

class IndexingPipeline:

    def __init__(self,loader: DocumentLoader,embedding_model: BaseEmbedding,vector_db: BaseVectorDB):
        self.loader = loader
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def index_document(self, file_path: str, user_id: str, session_id: str):
        chunks = self.loader.load(file_path)
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            embedding = (self.embedding_model.embed(chunk["text"]))
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            embeddings.append(embedding)
            metadatas.append({**chunk["metadata"], "user_id": user_id, "session_id": session_id})
        self.vector_db.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )