from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from app import app

with app.app_context():
    from services.rag import hybrid_retrieve, rerank_documents
    from extensions import reranker
    from flashrank import RerankRequest

    question = "what is CSMA and how does it work"

    print(f"🔍 Question: {question}\n")

    # Step 1: raw hybrid retrieval (before reranking)
    docs = hybrid_retrieve(question)
    print(f"📥 Retrieved {len(docs)} candidate chunks from hybrid retrieval (BM25 + dense)\n")

    print("=" * 80)
    print("BEFORE RERANKING (retrieval order)")
    print("=" * 80)
    for i, doc in enumerate(docs[:10], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"{i}. [{source}] {preview}...")

    # Step 2: show raw FlashRank scores directly
    print("\n" + "=" * 80)
    print("RAW RERANKER SCORES (FlashRank cross-encoder)")
    print("=" * 80)
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = reranker.rerank(rerank_request)
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    for r in ranked[:10]:
        idx = r["id"]
        score = r["score"]
        source = docs[idx].metadata.get("source", "unknown")
        preview = docs[idx].page_content[:80].replace("\n", " ")
        print(f"Score: {score:.4f} | [{source}] {preview}...")

    # Step 3: final reranked + filtered output (what actually reaches generation)
    final_docs = rerank_documents(question, docs, top_n=6)
    print("\n" + "=" * 80)
    print(f"AFTER RERANKING (final top_n={len(final_docs)} passed to generation)")
    print("=" * 80)
    for i, doc in enumerate(final_docs, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"{i}. [{source}] {preview}...")