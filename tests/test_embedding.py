from embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)


def main():

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    text = (
        "Artificial Intelligence is transforming healthcare."
    )

    vector = embedding_model.embed(text)

    print(
        f"Embedding dimension: {len(vector)}"
    )

    print(
        f"First 5 values: {vector[:5]}"
    )


if __name__ == "__main__":
    main()