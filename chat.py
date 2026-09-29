import json
import os
import urllib.request

MODEL = "fsociety-memory:latest"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MEMORY_FILE = "MEMORY.md"
HISTORY_FILE = "CHAT_HISTORY.jsonl"

RECENT_TURNS = 8


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return ""

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_all_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    rows = []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def load_recent_history():
    rows = load_all_history()
    return rows[-(RECENT_TURNS * 2):]


def save_message(role, content):
    item = {
        "role": role,
        "content": content
    }

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_messages(user_text, recent_history, memory):
    system_text = """
You are fsociety.

You are running inside a persistent long-term conversation system.

LONG-TERM MEMORY is background context.
Use it silently to maintain continuity.

IMPORTANT:
- Do not print or dump LONG-TERM MEMORY unless explicitly asked.
- Do not repeat old information unnecessarily.
- Do not restart the conversation from the beginning.
- Do not ask for information already available in memory or recent history.
- Do not claim memory was saved unless the controller actually saved it.
- Respond naturally and directly.

LONG-TERM MEMORY:
"""

    if memory:
        system_text += "\n" + memory
    else:
        system_text += "\nNo persistent memories stored yet."

    messages = [
        {
            "role": "system",
            "content": system_text
        }
    ]

    messages.extend(recent_history)

    messages.append({
        "role": "user",
        "content": user_text
    })

    return messages


def call_ollama(messages, think=False, stream=True, num_predict=1024):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "think": think,
        "options": {
            "num_ctx": 12288,
            "num_predict": num_predict
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    if stream:
        response_text = ""

        with urllib.request.urlopen(request) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = obj.get("message", {}).get("content", "")

                if chunk:
                    print(chunk, end="", flush=True)
                    response_text += chunk

                if obj.get("done"):
                    break

        print()
        return response_text

    else:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
            obj = json.loads(raw)

        return obj.get("message", {}).get("content", "")


def update_long_term_memory():
    current_memory = load_memory()
    history = load_all_history()

    if not history:
        print("No chat history to save.")
        return

    conversation_text = []

    for item in history:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        conversation_text.append(f"{role.upper()}: {content}")

    prompt = f"""
You are updating persistent long-term memory for this SAME ongoing project/chat.

CURRENT LONG-TERM MEMORY:

{current_memory}

NEW CHAT HISTORY:

{chr(10).join(conversation_text)}

Create an UPDATED long-term memory.

Rules:
- Preserve important existing memory unless newer information replaces it.
- Keep durable facts, goals, decisions, completed work, failures, current status,
  unresolved issues, and next steps.
- Remove repetition and temporary small talk.
- Do not include chain-of-thought or meta-commentary.
- Do not claim something is confirmed unless supported by the conversation.
- Keep it compact enough to load into future chats.
- Output only the updated memory document.

Use this structure:

# Long-Term Memory

## Main Goal

## Important Background

## Decisions Made

## Completed Work

## Current Status

## Unresolved / Unfinished Work

## Next Steps
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    print("\nUpdating long-term memory...")

    updated_memory = call_ollama(
        messages,
        think=False,
        stream=False,
        num_predict=1400
    ).strip()

    if not updated_memory:
        print("Memory update failed: empty response.")
        return

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(updated_memory + "\n")

    print("Long-term memory saved.")


def main():
    print(f"Model: {MODEL}")
    print("Long-term chat started.")
    print("Commands: /save, /exit")
    print()

    memory = load_memory()

    while True:
        try:
            user_text = input("You > ").strip()

        except KeyboardInterrupt:
            print("\nUse /exit to save memory before closing.")
            continue

        except EOFError:
            print("\nExiting.")
            break

        if not user_text:
            continue

        if user_text.lower() == "/save":
            update_long_term_memory()
            memory = load_memory()
            continue

        if user_text.lower() in {"/exit", "/quit"}:
            update_long_term_memory()
            print("Exiting.")
            break

        recent_history = load_recent_history()

        messages = build_messages(
            user_text=user_text,
            recent_history=recent_history,
            memory=memory
        )

        print("fsociety > ", end="", flush=True)

        try:
            assistant_text = call_ollama(
                messages,
                think=False,
                stream=True,
                num_predict=2048
            )

        except Exception as e:
            print(f"\nError: {e}")
            continue

        save_message("user", user_text)
        save_message("assistant", assistant_text)


if __name__ == "__main__":
    main()
