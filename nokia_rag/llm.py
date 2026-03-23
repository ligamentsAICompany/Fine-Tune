# ── nokia_rag/llm.py ────────────────────────────────────
# Handles prompt building and Ollama LLM calls

import ollama
import config

SYSTEM_PROMPT = """You are a Nokia ISAM technical assistant helping network engineers troubleshoot alarms and incidents.

Rules:
- Answer using ONLY the provided context.
- Be concise and technical — engineers don't need fluff.
- If the answer is not in the context, say exactly: "I could not find this in the Nokia documentation."
- For alarm-related questions, always mention: severity, root cause, and corrective action if available.
- Never hallucinate alarm codes, thresholds, or hardware specs."""


def build_user_message(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""CONTEXT FROM NOKIA ISAM DOCUMENTATION:
{context}

ENGINEER'S QUESTION:
{query}

ANSWER:"""


def generate_answer(
    query: str,
    context_chunks: list[str],
    history: list[dict],
) -> str:
    """
    Send query + retrieved context to Ollama.
    history is a list of prior {"role": ..., "content": ...} turns
    and is mutated in-place: the new user + assistant turn is appended.
    """
    user_content = build_user_message(query, context_chunks)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    response = ollama.chat(model=config.OLLAMA_MODEL, messages=messages)
    assistant_content = response["message"]["content"]

    # Append this turn to history — store only the raw query, NOT user_content.
    # Storing user_content (which contains all 5 context chunks) would cause
    # exponential context growth and context-window overflow after 3-4 turns.
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": assistant_content})

    return assistant_content
