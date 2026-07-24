from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import re

from app import app

with app.app_context():
    from services.intent_classifier import classify_intent

    test_questions = [
        "What does the mock caption say about the diagram in test_diagram_only_v2.pdf?",
        "What does the mock caption say about the diagram",
        "Summarize the diagram description found in test_diagram_only_v2.pdf",
        "explain the capstone project",  # known document_question example, sanity check
        "how are you",                    # known greeting example, sanity check
    ]

    print("=" * 70)
    print("INTENT CLASSIFIER RESULTS")
    print("=" * 70)
    for q in test_questions:
        category, score = classify_intent(q)
        print(f"Q: {q}")
        print(f"  -> category: {category} | confidence: {score:.4f}")
        print()

    # --- Test OLD vs NEW is_list_documents_query regex ---
    old_pattern = r"\b(list|show|what|which).*(document|file|pdf)s?\b"
    new_pattern = r"\b(list|show|what|which)\b.*(?<!\.)\b(document|file|pdf)s?\b"

    regex_test_cases = [
        ("What does the mock caption say about the diagram in test_diagram_only_v2.pdf?", False),
        ("list all documents", True),
        ("what documents do i have", True),
        ("show me the files", True),
        ("what files do you have", True),
        ("Summarize the diagram description found in test_diagram_only_v2.pdf", False),
        ("what does the CYBER_SECURITY_R18A0521.pdf say about firewalls", False),
    ]

    print("=" * 70)
    print("is_list_documents_query REGEX: OLD vs NEW")
    print("=" * 70)
    for question, expected in regex_test_cases:
        old_result = bool(re.search(old_pattern, question.lower().strip()))
        new_result = bool(re.search(new_pattern, question.lower().strip()))
        status = "✅" if new_result == expected else "❌"
        print(f"{status} Q: {question}")
        print(f"   expected: {expected} | OLD regex: {old_result} | NEW regex: {new_result}")
        print()