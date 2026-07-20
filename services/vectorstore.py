import os
import re
import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from extensions import embeddings
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

CHROMA_COLLECTION_NAME = "rag_documents"

def get_chroma_client():
    return chromadb.HttpClient(host="localhost", port=8000)

vectorstore = None
ALL_DOCS = []
dense_retriever = None
bm25_retriever = None

def initialize_vectorstore():
    global vectorstore, ALL_DOCS, dense_retriever, bm25_retriever
    try:
        client = get_chroma_client()
        vs = Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        existing = vs.get(include=["metadatas", "documents"])
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(existing.get("documents", []), existing.get("metadatas", []))
        ]
        valid_docs = [d for d in docs if d.metadata.get("source") != "system"]
        if valid_docs:
            print(f"📚 System successfully mapped {len(valid_docs)} document pieces into the runtime index context.")
            vectorstore = vs
            ALL_DOCS = valid_docs
            dense_retriever = vs.as_retriever(search_kwargs={"k": 10})
            bm25_retriever = BM25Retriever.from_documents(valid_docs, k=10)
            return vs, valid_docs
    except Exception as e:
        print(f"⚠️ Error loading vectorstore: {e}. Reinitializing.")

    print("⚠️ No valid document data discovered yet. Standing by for upload files...")
    dummy_doc = Document(
        page_content="No documents have been uploaded yet.",
        metadata={"source": "system", "filetype": "txt", "chunk_index": 0}
    )
    client = get_chroma_client()
    vs = Chroma.from_documents(
        [dummy_doc], embeddings,
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    vectorstore = vs
    ALL_DOCS = [dummy_doc]
    dense_retriever = vs.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents([dummy_doc], k=10)
    return vs, [dummy_doc]

def reload_vectorstore():
    global vectorstore, dense_retriever, bm25_retriever, ALL_DOCS
    initialize_vectorstore()
    print(f"🔄 Vectorstore configuration reloaded globally.")

# ============================================================
# --- FILE / SYNC HELPERS (moved from old app.py) ---
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_file_hash(filepath):
    stat = os.stat(filepath)
    return f"{stat.st_mtime_ns}:{stat.st_size}"

def get_file_version(filename):
    match = re.search(r'(_v\d+|_updated|_v\d+_updated)', filename.lower())
    return match.group(1) if match else "v1"

def sync_existing_documents():
    """Scan documents folder and add any files not in DB"""
    from extensions import db
    from models.models import User, ProcessedFile
    from ingest import ingest_documents
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return

        existing_files = os.listdir(UPLOAD_FOLDER)
        admin_user = User.query.filter_by(is_admin=True).first()
        if not admin_user:
            print("⚠️ No admin user found for syncing documents")
            return

        synced_count = 0
        for filename in existing_files:
            if not allowed_file(filename):
                continue

            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.isfile(filepath):
                continue

            existing = ProcessedFile.query.filter_by(filename=filename).first()
            if existing:
                continue

            try:
                file_hash = calculate_file_hash(filepath)
                file_size = os.path.getsize(filepath)
                version = get_file_version(filename)

                processed = ProcessedFile(
                    user_id=admin_user.id,
                    filename=filename,
                    file_hash=file_hash,
                    file_size=file_size,
                    chunk_count=0,
                    version=version
                )
                db.session.add(processed)
                synced_count += 1
            except Exception as e:
                print(f"⚠️ Error syncing {filename}: {e}")

        if synced_count > 0:
            db.session.commit()
            print(f"✅ Synced {synced_count} pre-existing documents to database")
            print("🔄 Running ingestion on synced documents...")
            ingest_documents()
            print("✅ Ingestion complete!")

    except Exception as e:
        print(f"❌ Sync error: {e}")                