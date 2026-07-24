import os
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from extensions import redis_client, count_tokens

# How many tokens of raw conversation to accumulate before triggering
# ONE infer=True (LLM) consolidation call, instead of one infer=True
# call per message.
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "500"))

_config = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "postgres",
            "password": "Bingos123",
            "host": "localhost",
            "port": "5432",
            "dbname": "mem0_db",
            "embedding_model_dims": 384,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "BX-cQI5qcf7Zj-i-bzJ_23dExiGQBGGDCHaaP6-WMQM",
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.3-70b-versatile",
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2"
        }
    }
}

memory = Memory.from_config(_config)


def get_user_memories(question, user_id, limit=5):
    """Read side — fetch relevant facts about this user for the current question.
    Unchanged: memory.search() is a vector similarity lookup, no LLM call."""
    try:
        result = memory.search(question, filters={"user_id": str(user_id)}, limit=limit)
        facts = [m["memory"] for m in result.get("results", [])]
        return "\n".join(f"- {f}" for f in facts) if facts else "No prior known facts about this user."
    except Exception:
        return "No prior known facts about this user."


def _window_text_key(user_id):
    return f"mem0_window_text:{user_id}"


def _window_ids_key(user_id):
    return f"mem0_window_ids:{user_id}"


def save_conversation_turn(question, answer, user_id):
    """
    Write side — sliding window approach (replaces the old always-infer add()):

    1. Store this turn as RAW text via infer=False — no LLM call, cheap and fast.
    2. Track the accumulated raw text + which memory IDs were written for it
       in Redis, keyed per user.
    3. Once the accumulated window's token count crosses MEMORY_TOKEN_LIMIT,
       run ONE infer=True call on the whole window so the LLM consolidates
       it into clean, durable facts — then delete the raw entries that were
       just summarized and reset the window.

    Net effect: instead of one LLM call per message, you get one LLM call
    only once a meaningful amount of conversation has accumulated.
    """
    turn_text = f"User: {question}\nAssistant: {answer}"

    # 1. Cheap raw write — no LLM call
    new_ids = []
    try:
        result = memory.add(
            [{"role": "user", "content": turn_text}],
            user_id=str(user_id),
            infer=False
        )
        new_ids = [r["id"] for r in result.get("results", []) if r.get("id")]
    except Exception as e:
        print(f"⚠️ mem0 raw add failed: {e}")

    # 2. Track this turn in the Redis sliding window
    text_key = _window_text_key(user_id)
    ids_key = _window_ids_key(user_id)
    try:
        redis_client.append(text_key, turn_text + "\n")
        if new_ids:
            redis_client.rpush(ids_key, *new_ids)
        # Safety TTL so an abandoned window doesn't linger forever
        redis_client.expire(text_key, 60 * 60 * 24)
        redis_client.expire(ids_key, 60 * 60 * 24)
    except Exception as e:
        print(f"⚠️ mem0 window tracking failed: {e}")
        return

    # 3. Check whether the window has exceeded the token limit
    window_text = redis_client.get(text_key) or ""
    token_count = count_tokens(window_text)

    if token_count >= MEMORY_TOKEN_LIMIT:
        _consolidate_memory_window(user_id, window_text, ids_key, text_key)


def _consolidate_memory_window(user_id, window_text, ids_key, text_key):
    """
    Runs ONE infer=True call so the LLM summarizes/cleans the accumulated
    raw window into proper long-term memory, then removes the raw
    (infer=False) entries it replaces and resets the window.
    """
    try:
        memory.add(
            [{"role": "user", "content": window_text}],
            user_id=str(user_id),
            infer=True
        )
        print(f"🧠 mem0 window consolidated for user {user_id} ({count_tokens(window_text)} tokens)")
    except Exception as e:
        print(f"⚠️ mem0 consolidation failed: {e}")
        # Don't clear the window if consolidation failed — retry on next turn
        # instead of silently losing the accumulated raw context.
        return

    # Delete the raw entries now that they're captured in the summary
    try:
        raw_ids = redis_client.lrange(ids_key, 0, -1)
        for mid in raw_ids:
            try:
                memory.delete(memory_id=mid)
            except Exception as del_err:
                print(f"⚠️ Could not delete raw memory {mid}: {del_err}")
    except Exception as e:
        print(f"⚠️ mem0 raw cleanup failed: {e}")

    # Reset the sliding window for this user
    redis_client.delete(text_key)
    redis_client.delete(ids_key)