import numpy as np
from extensions import embedder

CATEGORY_REFERENCES = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
        "how are you", "what can you do", "who are you",
        "thanks", "thank you", "okay", "cool", "bye", "goodbye"
    ],
    "personal_statement": [
        "I prefer dark mode over light mode",
        "I live in Kanpur",
        "my favorite color is blue",
        "I work at a tech company",
        "I was born in 2003",
        "I study electronics engineering",
        "I have a dog named Max"
    ],
    "document_question": [
        "what is the syllabus for this course",
        "explain the capstone project",
        "summarize chapter 3",
        "who are the members of the team",
        "what topics are covered in this module",
        "describe the cybersecurity framework"
    ]
}

_CATEGORY_EMBEDDINGS = {
    category: embedder.encode(phrases)
    for category, phrases in CATEGORY_REFERENCES.items()
}

def classify_intent(query, threshold=0.60):
    """
    Classifies a query into one of the categories using embedding
    similarity. Returns (category, confidence_score).
    If confidence is below threshold, falls back to 'document_question'
    (i.e. normal retrieval pipeline) — per issue #13's fallback requirement.
    """
    query_embedding = embedder.encode([query])[0]

    best_category = "document_question"
    best_score = -1.0

    for category, ref_embeddings in _CATEGORY_EMBEDDINGS.items():
        similarities = np.dot(ref_embeddings, query_embedding) / (
            np.linalg.norm(ref_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        max_sim = float(np.max(similarities))
        if max_sim > best_score:
            best_score = max_sim
            best_category = category

    if best_score >= threshold:
        return best_category, best_score
    else:
        return "document_question", best_score