"""
Quick retrieval test — asks a question and shows which chunks come back,
so you can confirm the diagram's VLM caption is actually retrievable.

Usage:
    python query_test.py "your question here"
"""
import sys
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "rag_documents"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What does the attack chain diagram show?"

    print(f"🔍 Query: {question}\n")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    results = vectorstore.similarity_search_with_score(question, k=5)

    if not results:
        print("⚠️ No results returned.")
        return

    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        print(f"--- Result {i} (distance: {score:.4f}) ---")
        print(f"Source: {meta.get('source')} | Page: {meta.get('page', 'N/A')} | Chunk: {meta.get('chunk_index')}")
        print(f"has_image_caption: {meta.get('has_image_caption', False)}")
        content = doc.page_content
        print(f"Content: {content[:400]}{'...' if len(content) > 400 else ''}")
        print()


if __name__ == "__main__":
    main()