def build_history(history: list[tuple]) -> str:
    if not history:
        return ""
    lines = []
    for q, a in history:
        if q == "Previous conversation summary":
            lines.append(f"[CONVERSATION SUMMARY]\n{a}")
        else:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)