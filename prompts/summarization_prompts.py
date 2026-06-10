def get_summarization_prompt(conversation: str) -> str:
    return f"""You are a conversation summarization assistant.

    RULES:
    1. Generate a clear, concise summary of the conversation below.
    2. Do not add any information not present in the conversation.
    3. Do not repeat information.

    CONVERSATION:{conversation}"""

def get_update_prompt(existing_summary: str, conversation: str) -> str:
    return (
        f"You are updating a rolling conversation summary.\n"
        f"Existing Summary:\n{existing_summary}\n\n"
        f"New Distinct Recent Interactions:\n{conversation}\n\n"
        f"Generate a brand new consolidated summary incorporating both without repeating info."
    )

def get_fresh_prompt(conversation: str) -> str:
    return f"Generate a clear, concise summary of this conversation:\n{conversation}"