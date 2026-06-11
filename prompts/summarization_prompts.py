def get_fresh_prompt(conversation: str) -> str:
    return (
        "You are a conversation summarizer.\n\n"
        "Summarize the following conversation into a concise memory block.\n"
        "Rules:\n"
        "- Preserve all specific entities mentioned (names, algorithms, scores, tables, values).\n"
        "- Keep track of what topics were discussed and what answers were given.\n"
        "- Write in third person: 'The user asked about X. The assistant explained Y.'\n"
        "- Be concise but do NOT lose factual details like numbers or names.\n\n"
        f"CONVERSATION:\n{conversation}"
    )


def get_update_prompt(existing_summary: str, new_conversation: str) -> str:
    return (
        "You are a conversation summarizer.\n\n"
        "You have an existing summary and new conversation turns.\n"
        "Merge them into one updated summary.\n"
        "Rules:\n"
        "- Keep all factual details from the existing summary.\n"
        "- Add new information from the new conversation.\n"
        "- Preserve specific entities (names, algorithms, scores, tables, values).\n"
        "- Remove redundancy but never drop facts.\n"
        "- Write in third person.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary}\n\n"
        f"NEW CONVERSATION:\n{new_conversation}"
    )


def get_summarization_prompt(context: str) -> str:
    """Generic summarization — used by SummarizationAgent.summarize()"""
    return (
        "Summarize the following content clearly and concisely.\n"
        "Preserve all key facts, numbers, names, and relationships.\n\n"
        f"CONTENT:\n{context}"
    )