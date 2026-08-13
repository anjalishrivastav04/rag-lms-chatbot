"""
worker/message_processor.py
---------------------------
Thin orchestrator for a single Kafka message — delegates all heavy
lifting to the clarification and result_store modules.
"""

import time

from logging_config import set_trace_id
from app import app
from services.cache import get_pending_parts
from worker.clarification import handle_pending_selection, handle_new_question
from worker.result_store import save_result, REDIS_RESULT_TTL


def _answer_question(question, session_id, user_id):
    """
    Wrapper that calls either the LangGraph pipeline or the plain
    RAG pipeline depending on the USE_LANGGRAPH env setting.
    Imported lazily here to keep the module usable even before the
    LangGraph graph is built.
    """
    import os
    USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

    if USE_LANGGRAPH:
        from worker.consumer_loop import get_graph
        graph = get_graph()
        result = graph.invoke({
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
        })
        return (
            result.get("answer", ""),
            result.get("cache_source", "NONE"),
            result.get("retrieval_info", ""),
            result.get("options", []),
            result.get("needs_escalation", False),
        )
    else:
        from services.rag import get_answer
        answer, cache_source, retrieval_info, options = get_answer(question, session_id)
        return answer, cache_source, retrieval_info, options, None


def process_message(payload: dict, logger) -> None:
    """
    Process a single chat-request payload end-to-end:
      1. Parse fields from the payload.
      2. Route to pending-selection OR new-question handler.
      3. Persist the result to Redis.
    """
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
            is_selection = (
                pending_parts
                and user_message.strip() in [str(i + 1) for i in range(len(pending_parts))]
            )

            if is_selection:
                result = handle_pending_selection(
                    user_id, session_id, request_id, user_message,
                    pending_parts, _answer_question, logger, start_time,
                )
            else:
                result = handle_new_question(
                    user_id, session_id, request_id, user_message,
                    _answer_question, logger, start_time,
                )

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
                "options": [],
            }

    save_result(request_id, result)
    logger.info("result_saved", extra={
        "req": request_id[:8],
        "ttl_s": REDIS_RESULT_TTL,
    })
