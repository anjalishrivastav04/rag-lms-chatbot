"""
worker/result_store.py
----------------------
Handles writing chat results (and timeout error payloads) to Redis
so the polling frontend can pick them up.
"""

import json
from extensions import redis_client

REDIS_RESULT_TTL = 300  # seconds


def save_result(request_id: str, result: dict) -> None:
    """Persist a completed result dict to Redis."""
    key = f"chat_result:{request_id}"
    redis_client.setex(key, REDIS_RESULT_TTL, json.dumps(result))


def timeout_result(request_id: str, timeout_seconds: int) -> None:
    """Write a timeout-error payload so the frontend stops polling."""
    key = f"chat_result:{request_id}"
    payload = {
        "reply": "⏰ Sorry, the AI took too long to respond. Please try again.",
        "retrieval_info": "",
        "cache_source": "TIMEOUT",
        "duration_ms": timeout_seconds * 1000,
        "parts": [],
        "options": [],
        "request_id": request_id,
    }
    redis_client.setex(key, REDIS_RESULT_TTL, json.dumps(payload))
