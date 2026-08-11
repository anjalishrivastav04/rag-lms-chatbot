import json
import time
import os
import threading

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv
from kafka_handler import get_consumer
from app import app
from extensions import redis_client
from services.rag import get_answer, save_chat_message, decompose_question, generate_followup_options, is_casual_query, is_list_documents_query
from services.evaluation import evaluate_response
from services.cache import get_pending_parts, save_pending_parts, clear_pending_parts
from logging_config import setup_logging, set_trace_id 
from services.vectorstore import initialize_vectorstore

load_dotenv()
REDIS_RESULT_TTL = 300
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
WORKER_TIMEOUT_SECONDS = 90  # ✅ Kill a hung LLM call after 90s

logger = setup_logging("worker")

if USE_LANGGRAPH:
    from graph_answer import build_graph
    _graph = None  # built once, lazily, on first use — see get_graph() below

    def get_graph():
        global _graph
        if _graph is None:
            _graph = build_graph()
            logger.info("langgraph_initialized")
        return _graph


_executor = ThreadPoolExecutor(max_workers=1)

def listen_for_vectorstore_updates():
    pubsub = redis_client.pubsub()
    pubsub.subscribe("vectorstore_updates")
    logger.info("vectorstore_reload_listener_started")
    for message in pubsub.listen():
        if message["type"] == "message":
            with app.app_context():
                initialize_vectorstore()
                logger.info("vectorstore_reloaded_via_signal")

def answer_question(question, session_id, user_id):
    if USE_LANGGRAPH:
        graph = get_graph()
        result = graph.invoke({
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
        })
        answer = result.get("answer", "")
        cache_source = result.get("cache_source", "NONE")
        retrieval_info = result.get("retrieval_info", "")
        options = result.get("options", [])
        needs_escalation = result.get("needs_escalation", False)
        confidence = result.get("confidence")
        logger.info("langgraph_answer_complete", extra={
            "session_id": session_id,
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "cache_source": cache_source,
        })
        return answer, cache_source, retrieval_info, options, needs_escalation
    else:
        answer, cache_source, retrieval_info, options = get_answer(question, session_id)
        return answer, cache_source, retrieval_info, options, None


def process_message(payload):
    request_id = payload["request_id"]
    user_id = payload["user_id"]
    session_id = payload["session_id"]
    user_message = payload["message"]

    set_trace_id(request_id)

    logger.info("message_processing_started", extra={
        "req": request_id[:8],
        "user": user_id,
        "session": session_id[:8],
        "q": user_message[:60],
    })
    start_time = time.time()

    with app.app_context():
        try:
            pending_parts = get_pending_parts(user_id)
            if pending_parts and user_message.strip() in [str(i+1) for i in range(len(pending_parts))]:
                selected_index = int(user_message.strip()) - 1
                selected = pending_parts[selected_index]
                remaining = [p for i, p in enumerate(pending_parts) if i != selected_index]

                if remaining:
                    save_pending_parts(user_id, remaining)
                else:
                    clear_pending_parts(user_id)

                reply, cache_source, retrieval_info, _, graph_escalation = answer_question(
                    selected["question"], session_id, user_id
                )

                skip_eval = is_casual_query(selected["question"]) or is_list_documents_query(selected["question"])
                if graph_escalation is not None:
                    # LangGraph already decided this — trust it, skip the old eval call
                    low_confidence = graph_escalation
                elif skip_eval:
                    low_confidence = False
                else:
                    score, feedback = evaluate_response(selected["question"], reply, retrieval_info, user_id, session_id)
                    no_info_found = "I don't have information about this" in reply
                    low_confidence = (not no_info_found) and (score is not None and score <= 2)

                if low_confidence:
                    reply += "\n\n⚠️ *I'm not fully confident in this answer — it may be incorrect.*"
                    logger.warning("low_confidence_answer", extra={
                        "request_id": request_id,
                        "question": selected["question"],
                    })

                options = [f"{i+1}. {p['label']}" for i, p in enumerate(remaining)] if remaining else generate_followup_options(selected["question"], reply)

                duration = int((time.time() - start_time) * 1000)
                save_chat_message(user_id, session_id, "assistant", reply,
                                  cache_source="CLARIFICATION", response_time_ms=duration)

                result = {
                    "reply": reply,
                    "retrieval_info": retrieval_info,
                    "cache_source": "CLARIFICATION",
                    "duration_ms": duration,
                    "options": options,
                    "parts": [{"question": selected["question"], "answer": reply, "low_confidence": low_confidence}],
                    "request_id": request_id
                }

            else:
                parts = decompose_question(user_message)

                if len(parts) > 1:
                    save_pending_parts(user_id, parts)
                    options = [f"{i+1}. {p['label']}" for i, p in enumerate(parts)]
                    clarification = "I found a few things in your question — what would you like to know first?"
                    duration = int((time.time() - start_time) * 1000)
                    save_chat_message(user_id, session_id, "assistant", clarification,
                                      cache_source="CLARIFICATION", response_time_ms=duration)
                    logger.info("question_decomposed", extra={
                        "request_id": request_id,
                        "num_parts": len(parts),
                    })
                    result = {
                        "reply": clarification,
                        "retrieval_info": "",
                        "cache_source": "CLARIFICATION",
                        "duration_ms": duration,
                        "options": options,
                        "parts": [],
                        "request_id": request_id
                    }

                else:
                    reply, cache_source, retrieval_info, options, graph_escalation = answer_question(
                        user_message, session_id, user_id
                    )

                    skip_eval = is_casual_query(user_message) or is_list_documents_query(user_message)
                    no_info_found = "I don't have information about this" in reply

                    if graph_escalation is not None:
                        # LangGraph already ran its own evaluate+retry loop — trust its verdict
                        low_confidence = graph_escalation
                    elif skip_eval:
                        low_confidence = False
                    else:
                        score, feedback = evaluate_response(user_message, reply, retrieval_info, user_id, session_id)
                        low_confidence = (not no_info_found) and (score is not None and score <= 2)

                    if low_confidence:
                        reply += "\n\n⚠️ *I'm not fully confident in this answer — it may be incorrect.*"
                        logger.warning("low_confidence_answer", extra={
                            "request_id": request_id,
                            "question": user_message,
                        })

                    duration = int((time.time() - start_time) * 1000)
                    save_chat_message(user_id, session_id, "assistant", reply,
                                      cache_source=cache_source, response_time_ms=duration)

                    result = {
                        "reply": reply,
                        "retrieval_info": retrieval_info,
                        "cache_source": cache_source,
                        "duration_ms": duration,
                        "options": options,
                        "parts": [{"question": user_message, "answer": reply, "low_confidence": low_confidence}],
                        "request_id": request_id
                    }

            logger.info("message_processing_completed", extra={
                "req": request_id[:8],
                "duration_ms": result["duration_ms"],
                "cache": result["cache_source"],
                "reply_len": len(result.get("reply", "")),
            })

        except Exception as e:
            logger.error("message_processing_failed", exc_info=True, extra={
                "req": request_id[:8],
                "user": user_id,
                "error": repr(e)[:120],
            })
            result = {
                "reply": f"Sorry, something went wrong: {repr(e)}",
                "retrieval_info": "",
                "cache_source": "NONE",
                "duration_ms": int((time.time() - start_time) * 1000),
                "parts": [],
                "options": []
            }

    result_key = f"chat_result:{request_id}"
    redis_client.setex(result_key, REDIS_RESULT_TTL, json.dumps(result))
    logger.info("result_saved", extra={
        "req": request_id[:8],
        "ttl_s": REDIS_RESULT_TTL,
    })


def main():
    from llm_provider import get_llm_provider
    logger.info("worker_starting", extra={
        "topic": "chat-requests",
        "llm_provider": get_llm_provider(),
        "using_langgraph": USE_LANGGRAPH,
        "timeout_s": WORKER_TIMEOUT_SECONDS,
    })

    with app.app_context():
        from services.vectorstore import initialize_vectorstore
        initialize_vectorstore()
        logger.info("vectorstore_initialized")

    reload_thread = threading.Thread(target=listen_for_vectorstore_updates, daemon=True)
    reload_thread.start()
    logger.info("vectorstore_reload_thread_started")

    consumer = get_consumer(group_id="rag-worker-group")

    try:
        logger.info("worker_consumer_loop_started")
        while True:
            try:
                # ✅ poll() returns after timeout_ms even with no messages,
                #    so Kafka's background heartbeat sender is never starved.
                #    The old `for message in consumer:` was a blocking iterator
                #    that prevented heartbeats during long LLM processing.
                records = consumer.poll(timeout_ms=5000)
                for tp, messages in records.items():
                    for message in messages:
                        payload = message.value
                        request_id = payload.get("request_id", "unknown")
                        future = _executor.submit(process_message, payload)
                        try:
                            future.result(timeout=WORKER_TIMEOUT_SECONDS)
                        except FuturesTimeoutError:
                            logger.error("worker_message_timeout", extra={
                                "request_id": request_id,
                                "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                            })
                            # ✅ Write a timeout error to Redis so the frontend stops polling
                            result_key = f"chat_result:{request_id}"
                            redis_client.setex(result_key, REDIS_RESULT_TTL, json.dumps({
                                "reply": "⏰ Sorry, the AI took too long to respond. Please try again.",
                                "retrieval_info": "",
                                "cache_source": "TIMEOUT",
                                "duration_ms": WORKER_TIMEOUT_SECONDS * 1000,
                                "parts": [],
                                "options": [],
                                "request_id": request_id,
                            }))
                        except Exception as exc:
                            logger.error("worker_message_processing_error", exc_info=True, extra={"request_id": request_id})
            except Exception as exc:
                logger.warning("worker_consumer_iteration_failed", exc_info=True, extra={"error": str(exc)})
                time.sleep(5)
    except KeyboardInterrupt:
        logger.info("worker_stopped_by_user")
    finally:
        consumer.close()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    main()