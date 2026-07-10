"""
Debug script — replicates the retrieval + rerank pipeline from
services/rag.py's get_answer(), but prints intermediate results at every
step (including actual FlashRank scores) instead of calling the LLM.

Run from your rag-chatbot project root, venv activated:

    python debug_retrieval.py "who is kanishka saxena"
"""

import sys
from app import app

def main(question):
    with app.app_context():
        from services.vectorstore import initialize_vectorstore
        initialize_vectorstore()

        from services.rag import (
            rag_fusion_retrieve,
            filter_person_docs_for_academic_query,
            is_academic_query,
        )
        from extensions import reranker
        from flashrank import RerankRequest

        print(f"\nQUESTION: {question}")
        print(f"is_academic_query: {is_academic_query(question)}")
        print("=" * 80)

        # Step 1: RAG Fusion retrieval (query variations + BM25 + dense + RRF)
        fused_docs = rag_fusion_retrieve(question)
        print(f"\n[STEP 1] After RAG Fusion + RRF: {len(fused_docs)} docs")
        for i, d in enumerate(fused_docs):
            src = d.metadata.get("source", "unknown")
            print(f"  {i+1}. {src}")

        # Step 2: person-doc filter (only fires for academic queries)
        filtered_docs = filter_person_docs_for_academic_query(question, fused_docs)
        print(f"\n[STEP 2] After person-doc filter: {len(filtered_docs)} docs (removed {len(fused_docs) - len(filtered_docs)})")

        # Step 3: rerank — but print RAW scores ourselves instead of just top_n
        print(f"\n[STEP 3] FlashRank scores for ALL {len(filtered_docs)} candidates:")
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(filtered_docs)]
        rerank_request = RerankRequest(query=question, passages=passages)
        results = reranker.rerank(rerank_request)
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)

        for r in ranked:
            idx = r["id"]
            src = filtered_docs[idx].metadata.get("source", "unknown")
            score = r["score"]
            preview = filtered_docs[idx].page_content[:100].replace("\n", " ")
            print(f"  score={score:.4f}  source={src}")
            print(f"      preview: {preview}...")

        print("\n" + "=" * 80)
        print("Look at the score GAP between rank 1-3 and the rest.")
        print("If top scores are all low (e.g. under ~0.3-0.5), it means NONE")
        print("of the retrieved chunks are genuinely relevant — the reranker")
        print("is just picking 'least bad' from a bad candidate pool. That")
        print("points back to retrieval (Step 1), not reranking, as the root cause.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python debug_retrieval.py "your question here"')
        sys.exit(1)
    main(sys.argv[1])