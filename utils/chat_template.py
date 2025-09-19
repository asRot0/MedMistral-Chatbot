from typing import List, Dict

def format_turns(history: List[Dict], user_input: str) -> str:
    lines = []
    for turn in history:
        r = turn.get("role", "user")
        c = turn.get("content", "").strip()
        prefix = "User" if r == "user" else "Assistant"
        lines.append(f"{prefix}: {c}")
    lines.append(f"User: {user_input.strip()}")
    return "\n".join(lines)
