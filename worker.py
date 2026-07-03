import json
import time
import os
from dotenv import load_dotenv
from kafka_handler import get_consumer

load_dotenv()
from logging_config import setup_logging, set_trace_id

logger = setup_logging("worker")
from app import app
from extensions import redis_client
from services.rag import get_answer, save_chat_message, decompose_question, generate_followup_options, is_casual_query, is_list_documents_query
from services.evaluation import evaluate_response
from services.cache import get_pending_parts, save_pending_parts, clear_pending_parts

# ✅ NEW: toggle between old if/else logic and the LangGraph version.
# Set USE_LANGGRAPH=true in your .env once you trust the graph in production.
# Defaults to "false" so nothing changes for you until you flip it on purpose.
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

if USE_LANGGRAPH:
    from graph_answer import build_graph
    _graph = None  # built once, lazily, on first use — see get_graph() below

    def get_graph():
        global _graph
        if _graph is None:
            _graph = build_graph()
            logger.info("langgraph_initialized")
        return _graph


REDIS_RESULT_TTL = 300

def answer_question(question, session_id, user_id):
    """
    Single entry point for getting an answer — routes to either the old
    get_answer() or the new LangGraph version, based on USE_LANGGRAPH.
    Returns the same 4-tuple shape either way, so callers don't need to care.
    """
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
        # Old path recomputes low_confidence itself further down (unchanged) —
        # return None here so process_message knows to use its existing logic.
        return answer, cache_source, retrieval_info, options, None


def process_message(payload):
    request_id = payload["request_id"]
    user_id = payload["user_id"]
    session_id = payload["session_id"]
    user_message = payload["message"]

    set_trace_id(request_id)

    logger.info("message_processing_started", extra={
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "message_preview": user_message[:50],
        "using_langgraph": USE_LANGGRAPH,
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
                "request_id": request_id,
                "duration_ms": result["duration_ms"],
                "cache_source": result["cache_source"],
            })

        except Exception as e:
            logger.error("message_processing_failed", exc_info=True, extra={
                "request_id": request_id,
                "user_id": user_id,
            })
            result = {
                "reply": f"Sorry, something went wrong: {str(e)}",
                "retrieval_info": "",
                "cache_source": "NONE",
                "duration_ms": int((time.time() - start_time) * 1000),
                "parts": [],
                "options": []
            }

    result_key = f"chat_result:{request_id}"
    redis_client.setex(result_key, REDIS_RESULT_TTL, json.dumps(result))
    logger.info("result_saved", extra={
        "request_id": request_id,
        "duration_ms": result["duration_ms"],
    })


def main():
    logger.info("worker_starting", extra={"topic": "chat-requests", "using_langgraph": USE_LANGGRAPH})

    with app.app_context():
        from services.vectorstore import initialize_vectorstore
        initialize_vectorstore()
        logger.info("vectorstore_initialized")

    consumer = get_consumer(group_id="rag-worker-group")

    try:
        for message in consumer:
            payload = message.value
            process_message(payload)
    except KeyboardInterrupt:
        logger.info("worker_stopped_by_user")
    finally:
        consumer.close()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    main()