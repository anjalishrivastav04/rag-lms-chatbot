"""
worker/clarification.py
-----------------------
Handles multi-part question decomposition and the pending-parts
selection flow: the two distinct branches that process_message must
choose between before it can generate an answer.
"""

import time

from services.rag import (
    decompose_question,
    save_chat_message,
    generate_followup_options,
)
from services.cache import get_pending_parts, save_pending_parts, clear_pending_parts
from worker.confidence import evaluate_confidence


LOW_CONFIDENCE_SUFFIX = "\n\n⚠️ *I'm not fully confident in this answer — it may be incorrect.*"


def handle_pending_selection(
    user_id: str,
    session_id: str,
    request_id: str,
    user_message: str,
    pending_parts: list,
    answer_fn,
    logger,
    start_time: float,
) -> dict:
    """
    The user typed a number to select one of several pending sub-questions.
    Answer that sub-question and carry the remaining parts forward.
    """
    selected_index = int(user_message.strip()) - 1
    selected = pending_parts[selected_index]
    remaining = [p for i, p in enumerate(pending_parts) if i != selected_index]

    if remaining:
        save_pending_parts(user_id, remaining)
    else:
        clear_pending_parts(user_id)

    reply, cache_source, retrieval_info, _, graph_escalation = answer_fn(
        selected["question"], session_id, user_id
    )

    low_confidence = evaluate_confidence(
        selected["question"], reply, retrieval_info, user_id, session_id, graph_escalation
    )

    if low_confidence:
        reply += LOW_CONFIDENCE_SUFFIX
        logger.warning("low_confidence_answer", extra={
            "request_id": request_id,
            "question": selected["question"],
        })

    options = (
        [f"{i+1}. {p['label']}" for i, p in enumerate(remaining)]
        if remaining
        else generate_followup_options(selected["question"], reply)
    )

    duration = int((time.time() - start_time) * 1000)
    save_chat_message(user_id, session_id, "assistant", reply,
                      cache_source="CLARIFICATION", response_time_ms=duration)

    return {
        "reply": reply,
        "retrieval_info": retrieval_info,
        "cache_source": "CLARIFICATION",
        "duration_ms": duration,
        "options": options,
        "parts": [{"question": selected["question"], "answer": reply, "low_confidence": low_confidence}],
        "request_id": request_id,
    }


def handle_new_question(
    user_id: str,
    session_id: str,
    request_id: str,
    user_message: str,
    answer_fn,
    logger,
    start_time: float,
) -> dict:
    """
    Brand-new question.  Decompose it first — if multiple sub-parts
    are found, ask the user to pick one; otherwise answer directly.
    """
    parts = decompose_question(user_message)

    # ── multi-part: ask user to clarify ──────────────────────────────
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
        return {
            "reply": clarification,
            "retrieval_info": "",
            "cache_source": "CLARIFICATION",
            "duration_ms": duration,
            "options": options,
            "parts": [],
            "request_id": request_id,
        }

    # ── single question: answer directly ─────────────────────────────
    reply, cache_source, retrieval_info, options, graph_escalation = answer_fn(
        user_message, session_id, user_id
    )

    low_confidence = evaluate_confidence(
        user_message, reply, retrieval_info, user_id, session_id, graph_escalation
    )

    if low_confidence:
        reply += LOW_CONFIDENCE_SUFFIX
        logger.warning("low_confidence_answer", extra={
            "request_id": request_id,
            "question": user_message,
        })

    duration = int((time.time() - start_time) * 1000)
    save_chat_message(user_id, session_id, "assistant", reply,
                      cache_source=cache_source, response_time_ms=duration)

    return {
        "reply": reply,
        "retrieval_info": retrieval_info,
        "cache_source": cache_source,
        "duration_ms": duration,
        "options": options,
        "parts": [{"question": user_message, "answer": reply, "low_confidence": low_confidence}],
        "request_id": request_id,
    }
