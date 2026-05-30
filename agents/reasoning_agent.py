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

                "Answer the question using only the information "
                "provided in the context.\n\n"

                "Do not use external knowledge.\n"

                "Do not make assumptions.\n"

                "If the answer is not present in the context, "
                "respond exactly:\n"

                "'I could not find the answer in the provided documents.'\n\n"

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