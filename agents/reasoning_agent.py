from llm.base_llm import BaseLLM
from llm.llm_config import LLMConfig
class ReasoningAgent:

    def __init__(self,llm: BaseLLM):
        self.llm = llm

    def answer(self, query: str, chunks: list[dict], conversation_history: list[tuple[str, str]], config: LLMConfig, summaries: list[tuple[str, str]] | None = None) -> str:
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
            context = "\n\n".join(chunk["text"] for chunk in chunks)
            summary_text = ""
            if summaries:
                summary_text = "\n\n".join(summary for summary, _ in summaries)
                summary_text = f"Summaries:\n{summary_text}\n\n"

            system_prompt = """
            You are an intelligent document question-answering assistant.
            Rules:
            1. Use ONLY the provided context.
            2. The context may contain:
                - document text
                - table summaries
                - table rows
            3. Compare the user question carefully with the context.
            4. Extract the required information accurately.
            5. When answering, use clear human language.
            6. Do not return raw chunks unless necessary.
            7. If the context contains table summaries or table rows,use them to answer accurately.
            8. Do not use external knowledge.
            9. Do not make assumptions.
            10. Present answers in natural language instead of copying raw context whenever possible.
            11. If the answer is not present in the context,respond exactly:

                I could not find the answer in the provided documents.
        """

            history_messages = []
            for query_text, answer_text in conversation_history:
                history_messages.append({"role": "user","content": query_text})
                history_messages.append({"role": "assistant","content": answer_text})
            messages = [{"role": "system","content": system_prompt}]
            if summary_text:
                messages.append({"role": "system","content": summary_text})
            messages.extend(history_messages)
            messages.append({"role": "user","content": (f"Context:\n{context}\n\n"f"Question:\n{query}")})
            return self.llm.generate(messages=messages, config=config)

        except Exception as e:
            raise RuntimeError("Failed to generate answer") from e