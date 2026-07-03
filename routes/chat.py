import uuid
import time
import json
import hashlib
import re
import threading
from flask import Blueprint, request, jsonify, session
from extensions import csrf, redis_client, embedder
from models.models import User
from services.cache import (
    get_cached_response, is_user_request_pending,
    set_user_request_pending, clear_user_request_pending,
    check_rate_limit, get_pending_parts, save_pending_parts,
    clear_pending_parts
)
from services.rag import (
    decompose_question, save_chat_message, get_answer
)
from services.evaluation import evaluate_response
from kafka_handler import send_chat_request, KAFKA_BROKER, KAFKA_TOPIC, KAFKA_ENABLED
from extensions import count_tokens
from config import TOKEN_LIMIT_MAX, RATE_LIMIT_WINDOW

chat_bp = Blueprint('chat', __name__)

# ============================================================
# --- CHAT ROUTE ---
# ============================================================

@chat_bp.route("/chat", methods=["POST"])
def chat():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"reply": "Please login first!", "error": "not_authenticated"})
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"reply": "User not found!", "error": "user_not_found"})

    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    client_session = data.get("session_id", "").strip()
    client_ip = request.remote_addr

    if not client_session or client_session == "default_session":
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"session-{user_id}"))
    else:
        session_id = client_session

    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    # ✅ PENDING PARTS SELECTION
    pending_parts = get_pending_parts(user_id)
    is_from_pending = False
    if pending_parts:
        stripped = user_message.strip()
        match = re.match(r'^(\d+)(?:[\.\s]|$)', stripped)
        if match:
            selected_index = int(match.group(1)) - 1
            if 0 <= selected_index < len(pending_parts):
                selected = pending_parts[selected_index]
                clear_pending_parts(user_id)
                user_message = selected["question"]
                is_from_pending = True
            else:
                clear_pending_parts(user_id)
        else:
            clear_pending_parts(user_id)

    # ✅ MULTI-PART DECOMPOSITION
    if not is_from_pending:
        parts = decompose_question(user_message)
        print(f"🔍 Decomposed into {len(parts)} parts: {parts}")
        if len(parts) > 1:
            save_pending_parts(user_id, parts)
            options = [f"{i+1}. {p['label']}" for i, p in enumerate(parts)]
            save_chat_message(user_id, session_id, "user", user_message, ip_address=client_ip)
            save_chat_message(user_id, session_id, "assistant",
                              "I found a few things in your question — what would you like to know first?",
                              cache_source="NONE")
            return jsonify({
                "reply": "I found a few things in your question — what would you like to know first?",
                "options": options,
                "cache_source": "CLARIFICATION",
                "done": True
            })
        user_message = parts[0]["question"]

    # ✅ CHECK PENDING REQUEST
    if is_user_request_pending(user_id):
        return jsonify({
            "reply": "⏳ Please wait — your previous request is still being processed.",
            "error": "request_pending"
        }), 429

    # ✅ RATE LIMIT
    token_count = count_tokens(user_message)
    allowed, remaining, reset_in = check_rate_limit(user_id, token_count)
    if not allowed:
        hours = reset_in // 3600
        mins = (reset_in % 3600) // 60
        return jsonify({
            "reply": f"⛔ You have reached the token limit of {TOKEN_LIMIT_MAX} tokens per 4-hour session. Please try again in {hours}h {mins}m.",
            "error": "rate_limit_exceeded",
            "reset_in": reset_in
        })

    try:
        from flask import current_app
        start_time = time.time()
        save_chat_message(user_id, session_id, "user", user_message, ip_address=client_ip)

        # ✅ TIER 0: Redis exact match
        exact_cache_key = f"user:{user_id}:exact:{hashlib.md5(user_message.encode()).hexdigest()}"
        redis_cached = redis_client.get(exact_cache_key)
        if redis_cached:
            duration = int((time.time() - start_time) * 1000)
            cached_data = json.loads(redis_cached)
            save_chat_message(user_id, session_id, "assistant", cached_data['reply'],
                              cache_source="REDIS_EXACT", response_time_ms=duration)
            def run_eval_redis():
                with current_app._get_current_object().app_context():
                    evaluate_response(user_message, cached_data['reply'],
                                      cached_data.get('retrieval_info', ''), user_id, session_id)
            threading.Thread(target=run_eval_redis, daemon=True).start()
            return jsonify({
                "reply": cached_data['reply'],
                "retrieval_info": cached_data.get('retrieval_info', ''),
                "cache_source": "REDIS_EXACT ⚡",
                "duration_ms": duration,
                "user": user.username,
                "options": [],
                "done": True
            })

        # ✅ TIER 1: Semantic cache
        query_embedding = embedder.encode(user_message)
        cached_response, content_type, similarity = get_cached_response(
            user_id, query_embedding, threshold=0.85)
        if cached_response:
            duration = int((time.time() - start_time) * 1000)
            save_chat_message(user_id, session_id, "assistant", cached_response,
                              cache_source="SEMANTIC_CACHE", response_time_ms=duration)
            def run_eval_semantic():
                with current_app._get_current_object().app_context():
                    evaluate_response(user_message, cached_response,
                                      f"Similarity: {similarity:.2%}", user_id, session_id)
            threading.Thread(target=run_eval_semantic, daemon=True).start()
            return jsonify({
                "reply": cached_response,
                "retrieval_info": f"Similarity: {similarity:.2%}",
                "cache_source": f"SEMANTIC_CACHE ({content_type}) 🧠",
                "duration_ms": duration,
                "user": user.username,
                "done": True
            })

        # ✅ TIER 2: Kafka queue
        request_id = send_chat_request(user_id, session_id, user_message)
        if not request_id:
            return jsonify({"reply": "❌ Failed to queue your request. Please try again.", "done": True})

        set_user_request_pending(user_id, request_id)
        return jsonify({
            "reply": "⏳ Processing your request...",
            "request_id": request_id,
            "done": False
        })

    except Exception as e:
        clear_user_request_pending(user_id)
        return jsonify({"reply": f"Error: {str(e)}", "done": True})


@csrf.exempt
@chat_bp.route("/chat/result/<request_id>", methods=["GET"])
def chat_result(request_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"done": False, "error": "not_authenticated"})
    result_key = f"chat_result:{request_id}"
    result = redis_client.get(result_key)
    if result:
        data = json.loads(result)
        redis_client.delete(result_key)
        clear_user_request_pending(user_id)
        return jsonify({**data, "done": True})
    return jsonify({"done": False})


@csrf.exempt
@chat_bp.route("/rate-limit/status", methods=["GET"])
def rate_limit_status():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "not_authenticated"}), 401
    try:
        key = f"rate_limit:{user_id}"
        count = redis_client.get(key)
        if count is None:
            return jsonify({
                "tokens_used": 0,
                "tokens_remaining": TOKEN_LIMIT_MAX,
                "limit": TOKEN_LIMIT_MAX,
                "window_hours": 4,
                "reset_in": RATE_LIMIT_WINDOW
            })
        count = int(count)
        ttl = redis_client.ttl(key)
        reset_in = ttl if ttl > 0 else RATE_LIMIT_WINDOW
        hours = reset_in // 3600
        mins = (reset_in % 3600) // 60
        return jsonify({
            "tokens_used": count,
            "tokens_remaining": max(0, TOKEN_LIMIT_MAX - count),
            "limit": TOKEN_LIMIT_MAX,
            "window_hours": 4,
            "reset_in": reset_in,
            "reset_in_str": f"{hours}h {mins}m"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@csrf.exempt
@chat_bp.route("/kafka/status", methods=["GET"])
def kafka_status():
    return jsonify({
        "kafka_enabled": KAFKA_ENABLED,
        "broker": KAFKA_BROKER,
        "topic": KAFKA_TOPIC,
        "mode": "Kafka Queue" if KAFKA_ENABLED else "Direct Processing"
    })


@chat_bp.route("/csrf-token", methods=["GET"])
def get_csrf_token():
    from flask_wtf.csrf import generate_csrf
    return jsonify({"csrf_token": generate_csrf()})