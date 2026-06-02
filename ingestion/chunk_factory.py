import uuid

class ChunkFactory:

    def create_text_chunk(self,text: str,metadata: dict) -> dict:

        if not text or not text.strip():
            raise ValueError("Chunk text cannot be empty")

        chunk_key = (f"{metadata.get('source', '')}:{text}")

        return {
            "chunk_id":str(uuid.uuid5(uuid.NAMESPACE_DNS,chunk_key)),
            "chunk_type": "text",
            "text": text,
            "metadata":{**metadata,"chunk_type": "text"} 
        }

    def create_table_chunk(self,summary: str,metadata: dict) -> dict:

        if not summary or not summary.strip():
            raise ValueError("Table summary cannot be empty")
        
        chunk_key = (f"{metadata.get('source', '')}:{summary}")

        return {
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS,chunk_key)),
            "chunk_type": "table",
            "text": summary,
            "metadata": {**metadata, "chunk_type": "table"}
        }

    def create_table_row_chunk(self,text: str,metadata: dict) -> dict:

        if not text or not text.strip():
            raise ValueError("Table row text cannot be empty")

        chunk_key = (f"{metadata.get('source', '')}:{text}")

        return {
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS,chunk_key)),
            "chunk_type": "table_row",
            "text": text,
            "metadata": {**metadata, "chunk_type": "table_row"}
        }