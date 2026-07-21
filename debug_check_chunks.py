# debug_check_chunks.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

for doc_id, doc in vs.docstore._dict.items():
    if doc.metadata.get("source") == "stock-vector-large-set-of-options-infographics-for-presentations-and-data-visualization-pyramid-timeline-1992034880.pdf":
        print("=" * 60)
        print("Chunk metadata:", doc.metadata)
        print("Chunk content:", doc.page_content[:100])
        print()