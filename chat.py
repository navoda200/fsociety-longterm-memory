import json
import os
import re
import urllib.request

MODEL = "fsociety-memory:latest"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MEMORY_FILE = "MEMORY.md"
HISTORY_FILE = "CHAT_HISTORY.jsonl"

# Keep newest 8 user/assistant turns exact in active context.
RECENT_TURNS = 8

# Consolidate when at least 8 older turns are waiting.
CONSOLIDATE_TURNS = 8

NORMAL_NUM_PREDICT = 8192
MEMORY_NUM_PREDICT = 4096
NUM_CTX = 24576


# ---------------------------------------------------------
# MEMORY
# ---------------------------------------------------------

def read_memory_file():
    if not os.path.exists(MEMORY_FILE):
        return "", 0

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    match = re.search(
        r"<!--\s*consolidated_messages:\s*(\d+)\s*-->",
        raw
    )

    consolidated_messages = int(match.group(1)) if match else 0

    # Do not inject controller metadata into the model.
    clean_memory = re.sub(
        r"<!--\s*consolidated_messages:\s*\d+\s*-->\s*",
        "",
        raw
    ).strip()

    return clean_memory, consolidated_messages


def write_memory(memory_text, consolidated_messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"<!-- consolidated_messages: "
            f"{consolidated_messages} -->\n"
        )
        f.write(memory_text.strip() + "\n")


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

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
                item = json.loads(line)

                if (
                    isinstance(item, dict)
                    and item.get("role") in {"user", "assistant"}
                    and isinstance(item.get("content"), str)
                ):
                    rows.append(item)

            except json.JSONDecodeError:
                continue

    return rows


def load_recent_history():
    rows = load_all_history()

    max_messages = RECENT_TURNS * 2

    return rows[-max_messages:]


def save_message(role, content):
    item = {
        "role": role,
        "content": content
    }

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# NORMAL CHAT CONTEXT
# ---------------------------------------------------------

def build_messages(user_text, recent_history, memory):
    system_text = """
You are fsociety.

You are running inside a persistent long-term conversation system.

LONG-TERM MEMORY is private background context.
Use it silently to maintain continuity.

IMPORTANT:
- Never dump LONG-TERM MEMORY unless explicitly asked.
- Do not repeat old information unnecessarily.
- Do not restart an ongoing project from the beginning.
- Do not ask for information already present in memory or recent chat.
- Do not claim memory was saved unless the controller actually saved it.
- RECENT CONVERSATION is more current than LONG-TERM MEMORY.
- If recent information conflicts with older memory, prefer the newer information.
- Respond naturally, directly, and accurately.

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


# ---------------------------------------------------------
# OLLAMA
# ---------------------------------------------------------

def call_ollama(
    messages,
    stream=True,
    num_predict=NORMAL_NUM_PREDICT
):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,

        # Fast normal operation.
        "think": False,

        "options": {
            "num_ctx": NUM_CTX,
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
        done_reason = None
        eval_count = 0

        with urllib.request.urlopen(request) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = obj.get("message", {}).get(
                    "content",
                    ""
                )

                if chunk:
                    print(chunk, end="", flush=True)
                    response_text += chunk

                if obj.get("done"):
                    done_reason = obj.get("done_reason")
                    eval_count = obj.get("eval_count", 0)
                    break

        print()

        # Detect replies that were stopped because the
        # generation/output limit was reached.
        if done_reason == "length":
            print()
            print(
                "[warning] Response reached the generation limit "
                "and may be incomplete."
            )
            print(
                "[warning] The model generated "
                f"{eval_count} tokens."
            )

        # Fallback for Ollama versions that may not
        # provide done_reason.
        elif done_reason is None and eval_count >= num_predict:
            print()
            print(
                "[warning] Response may have reached the "
                "generation limit."
            )

        return response_text

    with urllib.request.urlopen(request) as response:
       raw = response.read().decode("utf-8")
       obj = json.loads(raw)

    done_reason = obj.get("done_reason")
    eval_count = obj.get("eval_count", 0)

    if done_reason == "length":
       raise RuntimeError(
           f"Generation reached limit after {eval_count} tokens."
       )

    return obj.get("message", {}).get("content", "")


# ---------------------------------------------------------
# LONG-TERM MEMORY CONSOLIDATION
# ---------------------------------------------------------

def history_to_text(history):
    blocks = []

    for item in history:
        role = item["role"].upper()
        content = item["content"].strip()

        blocks.append(f"{role}:\n{content}")

    return "\n\n".join(blocks)


def consolidate_memory(force=False):
    history = load_all_history()
    memory, checkpoint = read_memory_file()

    # Never consolidate messages beyond history length.
    checkpoint = min(checkpoint, len(history))

    # Keep the newest RECENT_TURNS exact and unconsolidated.
    recent_message_count = RECENT_TURNS * 2

    eligible_end = max(
        0,
        len(history) - recent_message_count
    )

    if eligible_end <= checkpoint:
        if force:
            print("[memory] Nothing new to consolidate.")
        return False

    new_messages = history[checkpoint:eligible_end]

    threshold_messages = CONSOLIDATE_TURNS * 2

    if not force and len(new_messages) < threshold_messages:
        return False

    conversation_text = history_to_text(new_messages)

    memory_prompt = f"""
You maintain persistent long-term memory for one ongoing conversation.

CURRENT LONG-TERM MEMORY:
--- BEGIN CURRENT MEMORY ---
{memory if memory else "No previous long-term memory."}
--- END CURRENT MEMORY ---

NEW OLDER CONVERSATION TO CONSOLIDATE:
--- BEGIN NEW CONVERSATION ---
{conversation_text}
--- END NEW CONVERSATION ---

Create the UPDATED long-term memory.

The memory will be loaded into future chats after these exact messages
leave the recent conversation window.

KEEP:
- durable facts explicitly provided by the user
- ongoing project goals
- important requirements and preferences
- confirmed technical configuration
- decisions actually made
- useful assistant/model suggestions that may matter later
- useful ideas and alternatives discussed
- important open questions
- completed work
- failed attempts and lessons
- important filenames, model names, commands and architecture
- current project status
- unresolved problems
- relevant next steps

REMOVE:
- greetings
- repetition
- temporary filler
- obsolete details replaced by newer information
- chain-of-thought
- Thinking Process
- meta-commentary about summarizing memory
- duplicate information already represented more clearly elsewhere
- permanent model personality or system-prompt instructions unless directly relevant to the current project state

RULES:
- Preserve useful existing memory.
- Merge new information with existing memory.
- Newer confirmed information may replace outdated memory.
- Do not invent facts.
- Do not invent decisions.
- Do not invent project status.
- Do not invent future work.
- Do not claim planned work was completed.
- Distinguish user statements from confirmed technical results when needed.
- Keep useful assistant/model suggestions when they may matter later.
- Assistant/model suggestions must be stored under Suggestions, not Decisions.
- Hypothetical possibilities must be stored under Ideas.
- Open unresolved questions must be stored under Open Questions.
- Do not convert a Suggestion or Idea into a Decision unless the user accepts or confirms it.
- If the user later accepts a Suggestion, it may be promoted to a Decision.
- Do not treat assistant/model suggestions as user-confirmed facts.
- If a suggestion is rejected by the user, remove it or mark it as rejected only if that rejection remains useful.
- If an idea becomes obsolete, remove it.
- You may rephrase and compress information while preserving the original meaning.
- Keep the memory compact and useful.
- Do not describe this memory-update instruction.
- Do not include controller metadata such as consolidated_messages.
- Do not output <think> tags.
- Output only the memory document.

Use this structure:

# Long-Term Memory

## Main Goals
- Store confirmed long-term goals of the current project or conversation.

## Important Background
- Store durable context and background facts that may matter in future sessions.

## Requirements and Preferences
- Store important user requirements, constraints, preferences, and working style.

## Confirmed Environment
- Store confirmed models, software, files, configuration, architecture, operating system, versions, paths, and environment details when relevant.

## Decisions
- Store decisions explicitly made or accepted by the user.
- Do not place unconfirmed assistant suggestions here.

## Suggestions
- Store useful assistant/model recommendations that may matter later but have not yet been accepted as decisions.
- Keep the original purpose of each suggestion clear.
- If a suggestion is later accepted, move it to Decisions.
- If rejected or obsolete, remove it when appropriate.

## Ideas
- Store useful hypothetical approaches, alternatives, possible future features, and concepts worth remembering.
- Do not present ideas as confirmed plans.

## Completed Work
- Store work that was actually completed or clearly confirmed as finished.

## Failures and Lessons
- Store failed attempts, errors, lessons learned, and important corrections that should not be repeated.

## Current Status
- Store the latest confirmed state of the project.
- Prefer newer confirmed status over outdated status.

## Open Questions
- Store important questions that are still unresolved.
- Remove questions once they are answered unless the answer itself should be preserved elsewhere.

## Unresolved Work
- Store confirmed unfinished tasks, known problems, blockers, and work still remaining.

## Next Steps
- Store relevant next actions that are directly supported by the conversation.
- Do not invent next steps merely to fill this section.

Do not create content merely to fill a section.
If a section has no useful supported information, omit that section.

Output only the updated long-term memory document.
"""

    messages = [
        {
            "role": "user",
            "content": memory_prompt
        }
    ]

    print(
        f"\n[memory] Consolidating "
        f"{len(new_messages)} older messages..."
    )

    try:
        updated_memory = call_ollama(
            messages,
            stream=False,
            num_predict=MEMORY_NUM_PREDICT
        ).strip()

    except Exception as e:
        print(f"[memory] Update failed: {e}")
        return False

    if not updated_memory:
        print("[memory] Update failed: empty response.")
        return False

    # Only advance checkpoint AFTER successful memory generation.
    write_memory(
        updated_memory,
        eligible_end
    )

    print(
        f"[memory] Saved. "
        f"Checkpoint: {eligible_end} messages."
    )

    return True


# ---------------------------------------------------------
# STATUS
# ---------------------------------------------------------

def show_status():
    history = load_all_history()
    memory, checkpoint = read_memory_file()

    recent_count = min(
        len(history),
        RECENT_TURNS * 2
    )

    waiting = max(
        0,
        len(history)
        - (RECENT_TURNS * 2)
        - checkpoint
    )

    print()
    print("=== Memory Status ===")
    print(f"Model: {MODEL}")
    print(f"History messages: {len(history)}")
    print(f"Consolidated messages: {checkpoint}")
    print(f"Recent exact messages: {recent_count}")
    print(f"Waiting for consolidation: {waiting}")
    print(
        f"Memory characters: {len(memory)}"
    )
    print("=====================")
    print()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print(f"Model: {MODEL}")
    print("Long-term chat started.")
    print("Commands: /save, /status, /exit")
    print()

    while True:
        try:
            user_text = input("You > ").strip()

        except KeyboardInterrupt:
            print(
                "\nUse /exit to save memory before closing."
            )
            continue

        except EOFError:
            print("\nExiting.")
            break

        if not user_text:
            continue

        command = user_text.lower()

        if command == "/status":
            show_status()
            continue

        if command == "/save":
            consolidate_memory(force=True)
            continue

        if command in {"/exit", "/quit"}:
            consolidate_memory(force=True)
            print("Exiting.")
            break

        memory, _ = read_memory_file()
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
                stream=True,
                num_predict=NORMAL_NUM_PREDICT
            )

        except Exception as e:
            print(f"\nError: {e}")
            continue

        save_message("user", user_text)
        save_message("assistant", assistant_text)

        # Automatic memory consolidation.
        consolidate_memory(force=False)


if __name__ == "__main__":
    main()
