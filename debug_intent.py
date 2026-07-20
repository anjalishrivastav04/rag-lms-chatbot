"""
Quick debug script to see actual similarity scores per category
for a given query, using the embedding-based intent classifier.
"""
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from extensions import embedder
from services.intent_classifier import CATEGORY_REFERENCES, _CATEGORY_EMBEDDINGS

def debug_classify(query):
    query_embedding = embedder.encode([query])[0]
    print(f"\nQuery: {query!r}")
    print("-" * 60)

    scores = {}
    for category, ref_embeddings in _CATEGORY_EMBEDDINGS.items():
        similarities = np.dot(ref_embeddings, query_embedding) / (
            np.linalg.norm(ref_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        max_sim = float(np.max(similarities))
        best_phrase_idx = int(np.argmax(similarities))
        best_phrase = CATEGORY_REFERENCES[category][best_phrase_idx]
        scores[category] = max_sim
        print(f"{category:20s} -> max_sim={max_sim:.4f}  (closest ref: {best_phrase!r})")

    winner = max(scores, key=scores.get)
    print("-" * 60)
    print(f"Winner: {winner} (score={scores[winner]:.4f})")


if __name__ == "__main__":
    test_queries = [
        "I hate that the syllabus doesn't cover machine learning",
        "hi there",
        "I live in Kanpur",
        "tell me more",
    ]
    for q in test_queries:
        debug_classify(q)