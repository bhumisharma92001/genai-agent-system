from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
class ReasoningAgent:

    def __init__(
        self,
        llm: BaseLLM
    ):

        self.llm = llm

    def answer(
        self,
        query: str,
        chunks: list[dict],
        config: LLMConfig
    ) -> str:

        if not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if not chunks:

            return (
                "I could not find the answer "
                "in the provided documents."
            )

        try:

            context = "\n\n".join(
                chunk["text"]
                for chunk in chunks
            )

            prompt = (
                "You are a document question-answering assistant.\n\n"

                "Use only the information present in the provided context.\n\n"

                "The context may contain plain text, tables, spreadsheets, "
                "CSV records, Excel rows, or other structured data.\n\n"

                "Answer using the exact information found in the context.\n"

                "Return values directly when they are present in the context.\n"

                "If multiple records satisfy the question, return all relevant values.\n"

                "Do not use external knowledge.\n"

                "Do not make assumptions.\n\n"

                "If the answer is not present in the context, respond exactly:\n"

                "I could not find the answer in the provided documents.\n\n"

                f"Context:\n{context}\n\n"

                f"Question:\n{query}\n\n"

                "Answer:"
            )

            return self.llm.generate(
                prompt=prompt,
                config=config
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to generate answer"
            ) from e