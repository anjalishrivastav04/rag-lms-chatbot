import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # Dynamic Alpha Router
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

# 1. Setup Base Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the FAISS Index
print("💾 Loading FAISS Vector Store...")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# 3. Print Vectorstore Diagnostic Logs
all_docs = list(vectorstore.docstore._dict.values())
print(f"Total chunks in vectorstore: {len(all_docs)}")

print("\n--- Sources found ---")
sources = set(doc.metadata.get("source", "unknown") for doc in all_docs)
for s in sources:
    print(s)

# 4. Reconstruct Text Chunks for BM25 Keyword Search Engine
print("\n📝 Reconstructing text chunks for BM25 Keyword Search...")
faiss_docs = list(vectorstore.docstore._dict.values())

# 5. Initialize Individual Retrievers
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(faiss_docs)
bm25_retriever.k = 5

# 6. Combine Retrievers using the Alpha Weights Matrix
# alpha = 0.7 (70% Semantic Weight via FAISS, 30% Keyword Weight via BM25)
print("🎛️ Creating Hybrid Ensemble Retriever with Alpha = 0.7...")
alpha_hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # [FAISS Weight, BM25 Weight]
)

# 7. Apply Cross-Encoder Contextual Reranking Layout
print("🚀 Initializing Contextual Re-ranker (ms-marco-MiniLM-L-12-v2)...")
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=alpha_hybrid_retriever
)

# 8. Testing Retrieval
print("\n--- Testing Alpha Hybrid Retrieval & Reranking ---")
query = "education qualification degree"
docs = compression_retriever.invoke(query)

for idx, doc in enumerate(docs):
    print(f"\n[Rank {idx + 1}] Source: {doc.metadata.get('source')}")
    print(doc.page_content[:200] + "...")
# 8. Testing Retrieval with Alpha Breakdown
print("\n--- Testing Alpha Hybrid Retrieval & Reranking ---")
query = "education qualification degree"

# Fetch raw results from individual components to prove alpha behavior
faiss_raw = faiss_retriever.invoke(query)
bm25_raw = bm25_retriever.invoke(query)
final_reranked = compression_retriever.invoke(query)

print("\n🔍 [DEBUG] How Alpha Parameter Processed This Query:")
print(f"  • Alpha is set to 0.7 (70% FAISS Semantic / 30% BM25 Keyword)")
print(f"  • FAISS (Semantic) independently found {len(faiss_raw)} chunks matching the 'meaning' of the query.")
print(f"  • BM25 (Keyword) independently found {len(bm25_raw)} chunks matching exact words.")

print("\n🎯 Final Top Reranked Results After Hybrid Blending:")
for idx, doc in enumerate(final_reranked[:3]):  # Show top 3
    source_file = doc.metadata.get('source', 'Unknown')
    content_snippet = doc.page_content[:150].replace('\n', ' ')
    
    # Check which engine originally caught it
    in_faiss = any(doc.page_content == f.page_content for f in faiss_raw)
    in_bm25 = any(doc.page_content == b for b in bm25_raw)
    
    match_type = "Hybrid (Both)" if (in_faiss and in_bm25) else ("FAISS (Semantic)" if in_faiss else "BM25 (Keyword)")
    
    print(f"\n[Rank {idx + 1}] Source: {source_file} | Captured By: {match_type}")
    print(f"   ↳ Text: {content_snippet}...")