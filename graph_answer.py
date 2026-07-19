"""
graph_answer.py
----------------
LangGraph version of get_answer() from services/rag.py.
Now includes mem0 integration: recalls known facts about the user before
generating an answer, and saves new facts after the conversation turn.
"""

import logging
import threading
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

from services.rag import (
    is_casual_query,
    is_list_documents_query,
    is_personal_statement,
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
from services.memory_service import get_user_memories, save_conversation_turn
from graph_handler import graph_retrieve
from extensions import llm
from models.models import ProcessedFile

logger = logging.getLogger("worker")

MAX_RETRIES = 1
ESCALATE_THRESHOLD = 2
RETRY_THRESHOLD = 3


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
    memory_context: str
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
    from services.rag import is_confirmation_or_statement
    logger.info("graph_classify", extra={
        "session_id": state["session_id"],
        "question": question[:80],
    })
    return {
        "is_casual": is_casual_query(question) or is_personal_statement(question) or is_confirmation_or_statement(question),
        "is_list_query": is_list_documents_query(question),
        "original_question": question,
        "retry_count": 0,
    }


def recall_memory_node(state: ChatState) -> dict:
    facts = get_user_memories(state["question"], state["user_id"])
    logger.info("graph_memory_recalled", extra={
        "session_id": state["session_id"],
        "user_id": state["user_id"],
    })
    return {"memory_context": facts}


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
            if not any(
                bl.lower().rsplit('.', 1)[0] in doc.metadata.get("source", "").lower()
                for bl in blacklist
            )
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

    return {
        "docs": docs,
        "context": context + f"\n### KNOWLEDGE GRAPH:\n{graph_context}",
        "retrieval_info": retrieval_info
    }


def generate_node(state: ChatState) -> dict:
    question = state["question"]
    past_history = get_past_chat_history(state["session_id"], limit=6)
    context = state.get("context", "")
    docs = state.get("docs", [])

    # ✅ Check docs directly — context may have graph text even with no docs
    no_useful_docs = not docs or (
        len(docs) == 1 and docs[0].metadata.get("source") == "system"
    )

    if state.get("is_casual"):
        prompt = f"""You are RagBot, a friendly document assistant. Respond warmly and briefly.

### WHAT YOU KNOW ABOUT THIS USER:
{state.get('memory_context', 'No prior known facts about this user.')}
Use only the most recent fact if there are multiple versions. Do NOT narrate or reference what you previously knew — just respond naturally, as if you always knew this.

### QUESTION:
{question}
Answer:"""

    elif no_useful_docs:
        # ✅ No docs found — fall back to general LLM knowledge
        print("💡 No relevant docs found — falling back to LLM general knowledge")
        logger.info("graph_general_knowledge_fallback", extra={
            "session_id": state["session_id"],
            "question": question[:80],
        })
        prompt = f"""You are a helpful AI assistant with broad general knowledge.
The user asked a question that is not covered in the uploaded documents.
Answer it clearly and concisely using your general knowledge.
At the end, add a note: "ℹ️ This answer is from general knowledge, not from uploaded documents."

### PAST CONVERSATION:
{past_history}

### WHAT YOU KNOW ABOUT THIS USER:
{state.get('memory_context', 'No prior known facts about this user.')}

### QUESTION:
{question}

Answer:"""
        response = safe_invoke(llm, prompt)
        logger.info("graph_answer_generated", extra={
            "session_id": state["session_id"],
            "retry_count": state.get("retry_count", 0),
        })
        return {"answer": response.content, "cache_source": "GENERAL_LLM"}

    else:
        logger.info("graph_context_debug", extra={
        "session_id": state["session_id"],
        "doc_sources": [d.metadata.get("source") for d in docs],
        "context_preview": context[:1500],
    })
        # ✅ Docs found — answer from context
        prompt = f"""Answer strictly from the context below. If not present, say
"I don't have information about this in the uploaded documents."

### WHAT YOU KNOW ABOUT THIS USER:
{state.get('memory_context', 'No prior known facts about this user.')}
Use only the most recent fact if there are multiple versions of the same information. Do NOT narrate, reference, or explain what you previously said or didn't know — just answer the current question directly and naturally, as if you always knew this.

### PAST CONVERSATION:
{past_history}

### CONTEXT:
{context}

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
        return {"confidence": 5.0}

    if state.get("cache_source") == "GENERAL_LLM":
        return {"confidence": 5.0}

    answer = state.get("answer", "")

    # ✅ CHANGED: previously only forced confidence=0 when docs were
    # completely empty (no_docs). But the retriever almost always returns
    # *something* — even if totally irrelevant to the question — so
    # no_docs was rarely True in practice, and genuinely out-of-scope
    # questions (e.g. "free tools for static analysis") were falling
    # through to retry/escalate instead of the general-knowledge fallback,
    # even though the LLM had already clearly said it had no info.
    # Now we trust the LLM's own "I don't have information" signal
    # regardless of how many (irrelevant) docs were retrieved.
    if "I don't have information" in answer or "don't have information" in answer.lower():
        return {"confidence": 0}

    try:
        score, _ = evaluate_response(
            state["original_question"], state["answer"],
            state.get("retrieval_info", ""),
            state["user_id"], state["session_id"]
        )
        logger.info("graph_evaluated", extra={
            "session_id": state["session_id"],
            "confidence": score,
            "retry_count": state.get("retry_count", 0),
        })
        return {"confidence": score}
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e} — defaulting confidence to 3")
        return {"confidence": 3}

def retry_node(state: ChatState) -> dict:
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

    if (
        "[FileID:" not in answer and
        state.get("cache_source") != "REDIS" and
        state.get("cache_source") != "SEMANTIC"
    ):
        save_to_redis_cache(question, answer)
        save_to_semantic_cache(question, answer)

    return {"options": options}


def save_memory_node(state: ChatState) -> dict:
    def _save():
        save_conversation_turn(
            state["original_question"], state["answer"], state["user_id"]
        )
    threading.Thread(target=_save, daemon=True).start()
    return {}


def escalate_flag_node(state: ChatState) -> dict:
    logger.warning("graph_needs_escalation", extra={
        "session_id": state["session_id"],
        "confidence": state.get("confidence"),
        "question": state["original_question"][:80],
    })
    return {"needs_escalation": True}


def general_knowledge_node(state: ChatState) -> dict:
    question = state["original_question"]
    past_history = get_past_chat_history(state["session_id"], limit=6)
    print("💡 Answer not in docs — falling back to LLM general knowledge")
    logger.info("graph_general_knowledge_fallback", extra={
        "session_id": state["session_id"],
        "question": question[:80],
    })
    prompt = f"""You are a helpful AI assistant with broad general knowledge.
The user asked a question that is not covered in the uploaded documents.
Answer it clearly and concisely using your general knowledge.
At the end, add a note: "ℹ️ This answer is from general knowledge, not from uploaded documents."

### PAST CONVERSATION:
{past_history}

### QUESTION:
{question}

Answer:"""
    response = safe_invoke(llm, prompt)
    return {"answer": response.content, "cache_source": "GENERAL_LLM"}


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
    if state.get("cache_source") == "GENERAL_LLM":
        return "finalize"

    confidence = state.get("confidence")

    if confidence == 0:
        docs = state.get("docs", [])
        has_useful_docs = docs and not (
            len(docs) == 1 and docs[0].metadata.get("source") == "system"
        )
        # If real docs were retrieved but the LLM still claimed no info,
        # give it one reformulated retry before writing off the docs
        # entirely and falling back to general knowledge.
        if has_useful_docs and state.get("retry_count", 0) < MAX_RETRIES:
            return "retry"
        return "general_knowledge"

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
    graph.add_node("recall_memory", recall_memory_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("save_memory", save_memory_node)
    graph.add_node("escalate_flag", escalate_flag_node)
    graph.add_node("general_knowledge", general_knowledge_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "recall_memory")
    graph.add_conditional_edges("recall_memory", route_after_classify, {
        "generate": "generate", "cache_check": "cache_check",
    })
    graph.add_conditional_edges("cache_check", route_after_cache, {
        "finalize": "finalize", "retrieve": "retrieve",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {
        "finalize": "finalize",
        "retry": "retry",
        "escalate_flag": "escalate_flag",
        "general_knowledge": "general_knowledge",
    })
    graph.add_edge("retry", "retrieve")
    graph.add_edge("escalate_flag", "finalize")
    graph.add_edge("general_knowledge", "finalize")
    graph.add_edge("finalize", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()