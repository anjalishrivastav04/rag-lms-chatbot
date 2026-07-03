"""
graph_answer.py
----------------
LangGraph version of get_answer() from services/rag.py.

This does NOT replace get_answer() yet — it's a standalone module you can
test independently before deciding whether to wire it into worker.py.

Key difference from your current get_answer(): after evaluate_node scores
the answer, if confidence is borderline (not terrible, not great), the
graph loops back to retrieve_node with a reformulated query BEFORE
escalating to a human. This is the self-correction loop we discussed —
it re-grounds against fresh retrieval each time rather than just asking
the LLM to "improve" its own answer blind, which is what keeps this safe
from compounding hallucination.

Usage (once you're ready to wire it in):
    from graph_answer import build_graph
    app = build_graph()
    result = app.invoke({
        "question": user_message,
        "session_id": session_id,
        "user_id": user_id,
    })
    # result["answer"], result["cache_source"], result["retrieval_info"], result["options"]
"""

import logging
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

# Reuse your actual functions — nothing here is reimplemented, just wrapped.
from services.rag import (
    is_casual_query,
    is_list_documents_query,
    hybrid_retrieve,
    rerank_documents,
    safe_invoke,
    generate_followup_options,
    get_past_chat_history,
)
from services.cache import (
    check_redis_cache, save_to_redis_cache,
    check_semantic_cache, save_to_semantic_cache,
    get_blacklist,
)
from services.evaluation import evaluate_response
from graph_handler import graph_retrieve
from extensions import llm
from models.models import ProcessedFile

logger = logging.getLogger("worker")

MAX_RETRIES = 1  # how many times to self-correct before escalating
ESCALATE_THRESHOLD = 2  # confidence <= this -> escalate
RETRY_THRESHOLD = 3     # confidence <= this (but > ESCALATE_THRESHOLD) -> retry once


class ChatState(TypedDict):
    question: str
    session_id: str
    user_id: str
    original_question: str
    is_casual: bool
    is_list_query: bool
    docs: list
    context: str
    retrieval_info: str
    answer: str
    confidence: Optional[float]
    retry_count: int
    cache_source: str
    options: List[str]
    needs_escalation: bool


# ============================================================
# --- NODES ---
# ============================================================

def classify_node(state: ChatState) -> dict:
    question = state["question"]
    logger.info("graph_classify", extra={
        "session_id": state["session_id"],
        "question": question[:80],
    })
    return {
        "is_casual": is_casual_query(question),
        "is_list_query": is_list_documents_query(question),
        "original_question": question,
        "retry_count": 0,
    }


def cache_check_node(state: ChatState) -> dict:
    question = state["question"]
    cached = check_redis_cache(question)
    if cached:
        logger.info("graph_cache_hit", extra={"cache_source": "REDIS", "session_id": state["session_id"]})
        return {"answer": cached, "cache_source": "REDIS"}

    cached = check_semantic_cache(question)
    if cached:
        save_to_redis_cache(question, cached)
        logger.info("graph_cache_hit", extra={"cache_source": "SEMANTIC", "session_id": state["session_id"]})
        return {"answer": cached, "cache_source": "SEMANTIC"}

    return {"cache_source": "NONE"}


def retrieve_node(state: ChatState) -> dict:
    question = state["question"]
    docs = hybrid_retrieve(question)
    docs = rerank_documents(question, docs, top_n=6)

    blacklist = get_blacklist()
    if blacklist:
        docs = [
            doc for doc in docs
            if not any(bl.lower().rsplit('.', 1)[0] in doc.metadata.get("source", "").lower() for bl in blacklist)
        ]

    graph_context, related_entities = graph_retrieve(question)

    context = ""
    sources_set = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sources_set.add(source)
        file_record = ProcessedFile.query.filter_by(filename=source).first()
        file_id = file_record.file_id if file_record else "UNKNOWN"
        context += f"[FileID: {file_id}]\n{doc.page_content[:600]}\n\n"

    retrieval_info = f"📎 Sources: {', '.join(sources_set)}" if sources_set else ""

    logger.info("graph_retrieve_completed", extra={
        "session_id": state["session_id"],
        "num_docs": len(docs),
        "retry_count": state.get("retry_count", 0),
    })

    return {"docs": docs, "context": context + f"\n### KNOWLEDGE GRAPH:\n{graph_context}", "retrieval_info": retrieval_info}


def generate_node(state: ChatState) -> dict:
    question = state["question"]
    past_history = get_past_chat_history(state["session_id"], limit=6)

    if state.get("is_casual"):
        prompt = f"""You are RagBot, a friendly document assistant. Respond warmly and briefly.
### QUESTION:
{question}
Answer:"""
    else:
        prompt = f"""Answer strictly from the context below. If not present, say
"I don't have information about this in the uploaded documents."

### PAST CONVERSATION:
{past_history}

### CONTEXT:
{state.get('context', '')}

### QUESTION:
{question}

Answer:"""

    response = safe_invoke(llm, prompt)
    logger.info("graph_answer_generated", extra={
        "session_id": state["session_id"],
        "retry_count": state.get("retry_count", 0),
    })
    return {"answer": response.content}


def evaluate_node(state: ChatState) -> dict:
    if state.get("is_casual"):
        return {"confidence": 5.0}  # skip eval for casual, same as your current logic

    score, _ = evaluate_response(
        state["original_question"], state["answer"], state.get("retrieval_info", ""),
        state["user_id"], state["session_id"]
    )
    logger.info("graph_evaluated", extra={
        "session_id": state["session_id"],
        "confidence": score,
        "retry_count": state.get("retry_count", 0),
    })
    return {"confidence": score}


def retry_node(state: ChatState) -> dict:
    # ✅ Reformulate the query for the retry — re-grounds retrieval instead
    # of just asking the LLM to "improve" its previous answer blind.
    reformulated = f"{state['original_question']} (provide more specific detail)"
    logger.warning("graph_retry_triggered", extra={
        "session_id": state["session_id"],
        "attempt": state.get("retry_count", 0) + 1,
    })
    return {"question": reformulated, "retry_count": state.get("retry_count", 0) + 1}


def finalize_node(state: ChatState) -> dict:
    answer = state["answer"]
    question = state["original_question"]
    options = generate_followup_options(question, answer) if not state.get("is_casual") else []

    if "[FileID:" not in answer and state.get("cache_source") != "REDIS" and state.get("cache_source") != "SEMANTIC":
        save_to_redis_cache(question, answer)
        save_to_semantic_cache(question, answer)

    return {"options": options}


def escalate_flag_node(state: ChatState) -> dict:
    logger.warning("graph_needs_escalation", extra={
        "session_id": state["session_id"],
        "confidence": state.get("confidence"),
        "question": state["original_question"][:80],
    })
    return {"needs_escalation": True}


# ============================================================
# --- ROUTING ---
# ============================================================

def route_after_classify(state: ChatState) -> str:
    if state["is_casual"] or state["is_list_query"]:
        return "generate"
    return "cache_check"


def route_after_cache(state: ChatState) -> str:
    return "finalize" if state.get("cache_source") in ("REDIS", "SEMANTIC") else "retrieve"


def route_after_evaluate(state: ChatState) -> str:
    confidence = state.get("confidence")
    if confidence is None or confidence > RETRY_THRESHOLD:
        return "finalize"
    if confidence <= ESCALATE_THRESHOLD or state.get("retry_count", 0) >= MAX_RETRIES:
        return "escalate_flag"
    return "retry"


# ============================================================
# --- BUILD GRAPH ---
# ============================================================

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("escalate_flag", escalate_flag_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_after_classify, {
        "generate": "generate", "cache_check": "cache_check",
    })
    graph.add_conditional_edges("cache_check", route_after_cache, {
        "finalize": "finalize", "retrieve": "retrieve",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {
        "finalize": "finalize", "retry": "retry", "escalate_flag": "escalate_flag",
    })
    graph.add_edge("retry", "retrieve")  # the self-correction loop
    graph.add_edge("escalate_flag", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()