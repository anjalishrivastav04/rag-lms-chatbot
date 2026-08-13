"""
worker/confidence.py
--------------------
Single-responsibility module for evaluating whether an answer is
low-confidence enough to show a warning to the user.
"""

from services.rag import is_casual_query, is_list_documents_query
from services.evaluation import evaluate_response


def evaluate_confidence(
    question: str,
    reply: str,
    retrieval_info: str,
    user_id: str,
    session_id: str,
    graph_escalation,  # bool | None  — None means LangGraph was not used
) -> bool:
    """
    Return True if the answer should be flagged as low-confidence.

    Decision order:
    1. If LangGraph already evaluated the answer, trust its verdict.
    2. If the query is casual or a document-list request, never flag.
    3. Otherwise run the evaluator and flag when score <= 2
       (but not when the reply itself says "I don't have information").
    """
    skip_eval = is_casual_query(question) or is_list_documents_query(question)

    if graph_escalation is not None:
        return graph_escalation

    if skip_eval:
        return False

    score, _feedback = evaluate_response(question, reply, retrieval_info, user_id, session_id)
    no_info_found = "I don't have information about this" in reply
    return (not no_info_found) and (score is not None and score <= 2)
