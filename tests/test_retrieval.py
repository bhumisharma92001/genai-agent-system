from retrieval.retriever import Retriever


retriever = Retriever()

query = "What is accuracy score of Random forest and svm?"

results = retriever.retrieve(
    query=query,
    k=3
)

print("\nRETRIEVED CHUNKS:\n")

for index, result in enumerate(
    results,
    start=1
):

    print(
        f"\nCHUNK {index}\n"
    )

    print(
        result["text"]
    )

    print("\nMETADATA:\n")

    print(
        result["metadata"]
    )

    print(
        "\n" + "=" * 100
    )