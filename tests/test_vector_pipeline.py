from ingestion.document_loader import (
    DocumentLoader
)

from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)

from vector_db.chroma_db import (
    ChromaDB
)


# -----------------------------
# INITIALIZE COMPONENTS
# -----------------------------

loader = DocumentLoader()

embedding_model = (
    SentenceTransformerEmbedding()
)

vector_db = ChromaDB()


# -----------------------------
# LOAD DOCUMENT
# -----------------------------

chunks = loader.load(
    r"C:\Users\bhoomi.sharma\Downloads\advanced_ml_enterprise_report.pdf"
)

print("\nTOTAL CHUNKS:\n")

print(len(chunks))


# -----------------------------
# PREPARE VECTOR DATA
# -----------------------------

ids = []

documents = []

embeddings = []

metadatas = []


# -----------------------------
# PROCESS CHUNKS
# -----------------------------

for index, chunk in enumerate(chunks):

    print("\n" + "=" * 100)

    print(f"\nCHUNK {index + 1}\n")

    print("CHUNK ID:\n")

    print(chunk["chunk_id"])

    print("\nCHUNK TYPE:\n")

    print(chunk["chunk_type"])

    print("\nTEXT:\n")

    print(chunk["text"])

    print("\nMETADATA:\n")

    print(chunk["metadata"])

    # -----------------------------
    # GENERATE EMBEDDING
    # -----------------------------

    embedding = embedding_model.embed(
        chunk["text"]
    )

    print("\nEMBEDDING DIMENSION:\n")

    print(len(embedding))

    print("\nFIRST 10 EMBEDDING VALUES:\n")

    print(embedding[:10])

    # -----------------------------
    # STORE FOR VECTOR DB
    # -----------------------------

    ids.append(
        chunk["chunk_id"]
    )

    documents.append(
        chunk["text"]
    )

    embeddings.append(
        embedding
    )

    metadatas.append(
        chunk["metadata"]
    )


# -----------------------------
# UPSERT INTO CHROMADB
# -----------------------------

vector_db.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print("\n" + "=" * 100)

print(
    "\nDOCUMENT INDEXED SUCCESSFULLY\n"
)