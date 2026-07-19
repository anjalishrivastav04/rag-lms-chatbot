import os
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory

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
    """Read side — fetch relevant facts about this user for the current question."""
    try:
        result = memory.search(question, filters={"user_id": str(user_id)}, limit=limit)
        facts = [m["memory"] for m in result.get("results", [])]
        return "\n".join(f"- {f}" for f in facts) if facts else "No prior known facts about this user."
    except Exception:
        return "No prior known facts about this user."


def save_conversation_turn(question, answer, user_id):
    """Write side — extract and store durable facts from the user's message only."""
    try:
        memory.add(
            [{"role": "user", "content": question}],
            user_id=str(user_id)
        )
    except Exception:
        pass