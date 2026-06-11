import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- SETUP ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

ALL_DOCS = list(vectorstore.docstore._dict.values())
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)

RRF_K = 60

def reciprocal_rank_fusion(bm25_docs, dense_docs, alpha):
    """
    RRF with alpha parameter blending.
    
    Final Score = (alpha × FAISS) + ((1-alpha) × BM25)
    
    α=0.0 → Pure BM25 (keyword search)
    α=0.5 → Balanced hybrid
    α=1.0 → Pure FAISS (semantic search)
    """
    scores = {}
    doc_map = {}

    # BM25 scores
    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        bm25_score = 1.0 / (RRF_K + rank)
        scores[key] = scores.get(key, 0.0) + ((1 - alpha) * bm25_score)
        doc_map[key] = doc

    # FAISS scores
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content
        faiss_score = 1.0 / (RRF_K + rank)
        scores[key] = scores.get(key, 0.0) + (alpha * faiss_score)
        doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [(doc_map[k], scores[k]) for k in ranked_keys]

def test_alpha_values(question):
    """Test different alpha values and show results."""
    
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")

    # Get base retrievals
    bm25_docs = bm25_retriever.invoke(question)
    dense_docs = dense_retriever.invoke(question)

    # Test different alpha values
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]

    for alpha in alpha_values:
        print(f"\n{'─'*80}")
        print(f"α = {alpha} → ", end="")
        
        if alpha == 0.0:
            print("PURE BM25 (Keyword Search Only) 🔑")
        elif alpha == 1.0:
            print("PURE FAISS (Semantic Search Only) 🧠")
        else:
            print(f"HYBRID ({int((1-alpha)*100)}% BM25 + {int(alpha*100)}% FAISS) ⚖️")
        
        print(f"{'─'*80}\n")

        # Retrieve with this alpha
        results = reciprocal_rank_fusion(bm25_docs, dense_docs, alpha)

        # Show top 3 results
        for i, (doc, score) in enumerate(results[:3], 1):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            
            print(f"{i}. [{source} | Chunk {chunk_idx}]")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:100]}...")
            print()

    # Comparison table
    print(f"\n{'='*80}")
    print("ALPHA PARAMETER SUMMARY")
    print(f"{'='*80}\n")

    comparison = {
        "0.0": "Pure BM25 - Best for exact keywords, course codes, assignment names",
        "0.3": "30% Semantic + 70% Keyword - Slight semantic boost",
        "0.5": "Balanced - 50/50 split between meaning and keywords",
        "0.7": "70% Semantic + 30% Keyword - Our choice! Best for conversational student queries",
        "1.0": "Pure FAISS - Best for understanding meaning, but may miss exact terms"
    }

    for alpha_str, desc in comparison.items():
        print(f"α={alpha_str} → {desc}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Test with different questions
    questions = [
        "What are the technical skills required?",
        "Tell me about Python",
        "What is capstone project",
        "How to run the system"
    ]

    for q in questions:
        test_alpha_values(q)
        input("Press Enter for next question...")