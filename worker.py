"""
worker.py
Kafka Consumer Worker — runs separately from app.py.
Picks up ONE chat request at a time from the Kafka queue,
runs it through the existing RAG pipeline, and saves the
result to Redis so the Flask app can return it to the user.

Run this with:
    python worker.py

Keep this running in its own terminal alongside:
    - Kafka broker (kafka-server-start.bat)
    - Flask app (python app.py)
"""

import json
import time
import redis
import os
from dotenv import load_dotenv
from kafka_handler import get_consumer

load_dotenv()

# ============================================================
# --- IMPORTANT: import everything needed to run get_answer() ---
# We import app.py's Flask app + get_answer so we reuse the
# EXACT same RAG pipeline (vectorstore, caching, evaluation, etc.)
# without duplicating any logic.
# ============================================================

from app import app, get_answer, save_chat_message, evaluate_response, redis_client

REDIS_RESULT_TTL = 300  # Keep results in Redis for 5 minutes in case frontend is slow to poll


def process_message(payload):
    """Run one chat request through the full RAG pipeline and store the result."""
    request_id = payload["request_id"]
    user_id = payload["user_id"]
    session_id = payload["session_id"]
    user_message = payload["message"]

    print(f"⚙️  Processing request {request_id} for user {user_id}: '{user_message[:50]}'")

    start_time = time.time()

    with app.app_context():
        try:
            reply, cache_source, retrieval_info = get_answer(user_message, session_id)
            duration = int((time.time() - start_time) * 1000)

            # Save to chat history (same as before)
            save_chat_message(user_id, session_id, "assistant", reply,
                              cache_source=cache_source, response_time_ms=duration)

            # Run evaluation (same as before)
            evaluate_response(user_message, reply, retrieval_info, user_id, session_id)

            result = {
                "reply": reply,
                "retrieval_info": retrieval_info,
                "cache_source": cache_source,
                "duration_ms": duration
            }

        except Exception as e:
            print(f"❌ Error processing request {request_id}: {e}")
            result = {
                "reply": f"Sorry, something went wrong while processing your request: {str(e)}",
                "retrieval_info": "",
                "cache_source": "NONE",
                "duration_ms": int((time.time() - start_time) * 1000)
            }

    # ✅ Save result to Redis so Flask's /chat/result/<request_id> can pick it up
    result_key = f"chat_result:{request_id}"
    redis_client.setex(result_key, REDIS_RESULT_TTL, json.dumps(result))
    print(f"✅ Result saved for request {request_id} ({result['duration_ms']}ms)")


def main():
    print("🚀 Kafka Consumer Worker starting...")
    print("📡 Listening on topic: chat-requests")
    print("⚙️  Processing ONE message at a time (sequential, no concurrency)")
    print("-" * 60)

    consumer = get_consumer(group_id="rag-worker-group")

    try:
        for message in consumer:
            payload = message.value
            process_message(payload)
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()