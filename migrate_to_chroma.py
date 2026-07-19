"""
One-time migration: reads all documents out of the OLD FAISS vectorstore
folder and re-inserts them into the NEW ChromaDB collection, so existing
uploaded documents aren't lost when switching vector stores.

Run this ONCE, after installing langchain-chroma/chromadb, before you
start using the app again. Safe to delete afterward.
"""
import chromadb
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

OLD_FAISS_DIR = "vectorstore"
COLLECTION_NAME = "rag_documents"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("💾 Loading old FAISS index...")
old_vs = FAISS.load_local(OLD_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

docs = [old_vs.docstore.search(idx) for idx in old_vs.index_to_docstore_id.values()]
valid_docs = [d for d in docs if d.metadata.get("source") != "system"]

for d in valid_docs:
    d.metadata.setdefault("status", "active")

print(f"📦 Found {len(valid_docs)} real chunks to migrate (excluding placeholder docs).")

if valid_docs:
    print("🔄 Re-embedding and inserting into ChromaDB server (this runs locally, no API cost, just takes a moment)...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    new_vs = Chroma.from_documents(
        valid_docs, embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Migration complete — {len(valid_docs)} chunks now in ChromaDB server.")
else:
    print("⚠️ No real documents found in the old FAISS index — nothing to migrate.")