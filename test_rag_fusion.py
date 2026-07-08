from app import app
from services.rag import rag_fusion_retrieve, hybrid_retrieve, generate_query_variations, rerank_documents, filter_person_docs_for_academic_query

with app.app_context():
    from services.vectorstore import initialize_vectorstore
    initialize_vectorstore()
    print("📚 Vectorstore initialized for test.")

    q = "what topics are covered in the python syllabus"

    print("\n--- Query Variations Generated ---")
    variations = generate_query_variations(q)
    for v in variations:
        print(" -", v)

    print("\n--- OLD: hybrid_retrieve ---")
    for d in hybrid_retrieve(q)[:6]:
        print(d.metadata.get("source"), "-", d.page_content[:80])

    print("\n--- NEW: rag_fusion_retrieve ---")
    for d in rag_fusion_retrieve(q)[:6]:
        print(d.metadata.get("source"), "-", d.page_content[:80])

    print("\n--- rag_fusion_retrieve + reranked ---")
    fused = rag_fusion_retrieve(q)
    reranked = rerank_documents(q, fused, top_n=6)
    for d in reranked:
        print(d.metadata.get("source"), "-", d.page_content[:80])

    print("\n--- fusion + person-doc filter + reranked ---")
    fused = rag_fusion_retrieve(q)
    filtered = filter_person_docs_for_academic_query(q, fused)
    reranked = rerank_documents(q, filtered, top_n=6)
    for d in reranked:
        print(d.metadata.get("source"), "-", d.page_content[:80])

    print("\n--- non-academic query check: 'tell me about anjali' ---")
    q2 = "tell me about anjali"
    fused2 = rag_fusion_retrieve(q2)
    filtered2 = filter_person_docs_for_academic_query(q2, fused2)
    reranked2 = rerank_documents(q2, filtered2, top_n=6)
    for d in reranked2:
        print(d.metadata.get("source"), "-", d.page_content[:80])