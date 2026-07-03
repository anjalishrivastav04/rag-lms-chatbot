"""
graph_tools.py
---------------
LangGraph/LangChain "tools" for your RAG system — each wraps an existing
function from your codebase so it can be invoked either:
  (a) directly, as a node in the deterministic graph (graph_answer.py), or
  (b) dynamically, if you later let an LLM decide which tool to call
      (via .bind_tools() + ToolNode — see the bottom of this file).

Each tool needs:
  - the @tool decorator
  - a type-hinted signature (this becomes the tool's schema)
  - a clear docstring (the LLM reads this to decide when to use the tool)
"""

import logging
from langchain_core.tools import tool

from services.rag import hybrid_retrieve, rerank_documents
from services.cache import check_redis_cache, check_semantic_cache, get_blacklist
from graph_handler import graph_retrieve
from models.models import ProcessedFile

logger = logging.getLogger("worker")


@tool
def retrieve_documents(question: str) -> str:
    """Search the uploaded document knowledge base (FAISS + BM25 hybrid search)
    for chunks relevant to the question. Use this when the question is about
    content that might be in the uploaded documents."""
    docs = hybrid_retrieve(question)
    docs = rerank_documents(question, docs, top_n=6)

    blacklist = get_blacklist()
    if blacklist:
        docs = [
            d for d in docs
            if not any(bl.lower().rsplit('.', 1)[0] in d.metadata.get("source", "").lower() for bl in blacklist)
        ]

    logger.info("tool_retrieve_documents", extra={"question": question[:80], "num_docs": len(docs)})

    if not docs:
        return "No relevant documents found."

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        file_record = ProcessedFile.query.filter_by(filename=source).first()
        file_id = file_record.file_id if file_record else "UNKNOWN"
        formatted.append(f"[FileID: {file_id}] {doc.page_content[:600]}")
    return "\n\n".join(formatted)


@tool
def graph_lookup(question: str) -> str:
    """Query the Neo4j knowledge graph for entities and relationships related
    to the question. Use this for questions about how concepts/entities relate
    to each other, not just plain document text search."""
    graph_context, related_entities = graph_retrieve(question)
    logger.info("tool_graph_lookup", extra={"question": question[:80], "num_entities": len(related_entities or [])})
    return graph_context or "No related graph entities found."


@tool
def check_cache(question: str) -> str:
    """Check Redis and semantic cache for a previously-answered version of this
    question, to avoid redundant retrieval/generation. Returns the cached
    answer if found, or 'NO_CACHE_HIT' if nothing is cached."""
    cached = check_redis_cache(question) or check_semantic_cache(question)
    logger.info("tool_check_cache", extra={"question": question[:80], "hit": bool(cached)})
    return cached if cached else "NO_CACHE_HIT"


@tool
def list_available_documents() -> str:
    """List all documents currently available in the knowledge base (excludes
    deleted/blacklisted files). Use this when the user asks what documents
    exist or what files can be searched."""
    blacklist_filenames = set(get_blacklist())
    files = ProcessedFile.query.all()
    valid_files = [f for f in files if f.filename not in blacklist_filenames]
    logger.info("tool_list_documents", extra={"count": len(valid_files)})
    if not valid_files:
        return "No documents are currently available."
    return "\n".join(f"- {f.file_id}" for f in valid_files)


@tool
def escalate_to_admin(question: str, ai_answer: str, user_email: str) -> str:
    """Escalate a question to a human admin when confidence is too low to
    answer reliably. Sends an email to the admin with the question and the
    AI's best (low-confidence) attempt, and returns a resolution link."""
    import secrets
    from extensions import db, mail
    from models.models import EscalationRequest
    from config import ADMIN_EMAIL
    from flask_mail import Message as MailMessage

    token = secrets.token_urlsafe(32)
    escalation = EscalationRequest(
        token=token, user_email=user_email, question=question, ai_answer=ai_answer
    )
    db.session.add(escalation)
    db.session.commit()

    msg = MailMessage(subject="🚨 RAGBot Escalation", recipients=[ADMIN_EMAIL])
    msg.html = f"<p><b>Question:</b> {question}</p><p><b>AI Answer:</b> {ai_answer}</p>"
    mail.send(msg)

    logger.warning("tool_escalate_to_admin", extra={"escalation_id": escalation.id, "question": question[:80]})
    return f"Escalated successfully. Admin will review at token {token[:8]}..."


# ============================================================
# All tools, ready to bind to an LLM or wrap in a ToolNode
# ============================================================
ALL_TOOLS = [retrieve_documents, graph_lookup, check_cache, list_available_documents, escalate_to_admin]


# ============================================================
# OPTIONAL: how you'd use these if you wanted an LLM to pick
# dynamically, instead of your deterministic graph routing.
# Not required for your current setup — shown for reference.
# ============================================================
"""
from langgraph.prebuilt import ToolNode
from extensions import llm

llm_with_tools = llm.bind_tools(ALL_TOOLS)
tool_node = ToolNode(ALL_TOOLS)

# In a graph, you'd add this as a node, and route to it whenever the LLM's
# response contains a tool_call — this is the "agentic" pattern where the
# LLM itself decides which tool(s) to invoke, rather than your fixed routing
# functions (route_after_classify, route_after_evaluate, etc.) in graph_answer.py.
"""