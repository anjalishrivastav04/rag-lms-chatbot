from dotenv import load_dotenv
load_dotenv()

import os
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import app  # reuse existing Flask app instance for context

with app.app_context():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host="localhost", port=8000)
    vectorstore = Chroma(client=client, collection_name="rag_documents", embedding_function=embeddings)

    from services.cache import get_blacklist
    bl = get_blacklist()
    print(f"Current blacklist: {bl}")
    print()

    raw = vectorstore.get(where={"source": "test_diagram_only_v2.pdf"}, include=["metadatas"])
    metas = raw.get("metadatas", [])
    print(f"Found {len(metas)} chunks for test_diagram_only_v2.pdf")
    for m in metas:
        print(f"  status: {m.get('status')} | has_image_caption: {m.get('has_image_caption')}")