import os
import re
import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from extensions import embeddings
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

COLLECTION_NAME = "rag_documents"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

vectorstore = None
ALL_DOCS = []
dense_retriever = None
bm25_retriever = None

def is_not_archived(metadata: dict) -> bool:
    """Used for BM25 only (BM25 has no native metadata filtering, so it
    still needs a Python-side check). Dense retrieval no longer needs
    this — ChromaDB filters natively via a where-clause, during the
    search itself, instead of after."""
    return metadata.get("status", "active") != "archived"

def get_chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def initialize_vectorstore():
    global vectorstore, ALL_DOCS, dense_retriever, bm25_retriever

    try:
        vs = Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        raw = vs.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
        ]
        valid_docs = [d for d in docs if d.metadata.get("source") != "system"]

        if valid_docs:
            print(f"📚 System successfully mapped {len(valid_docs)} document pieces into the runtime index context.")
            vectorstore = vs
            ALL_DOCS = valid_docs
            dense_retriever = vs.as_retriever(
                search_kwargs={"k": 10, "filter": {"status": {"$ne": "archived"}}}
            )
            bm25_retriever = BM25Retriever.from_documents(valid_docs, k=10)
            return vs, valid_docs
    except Exception as e:
        print(f"⚠️ Error loading vectorstore: {e}. Reinitializing.")

    print("⚠️ No valid document data discovered yet. Standing by for upload files...")
    dummy_doc = Document(
        page_content="No documents have been uploaded yet.",
        metadata={"source": "system", "filetype": "txt", "chunk_index": 0, "status": "active"}
    )
    vs = Chroma.from_documents(
        [dummy_doc], embeddings,
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
    )
    vectorstore = vs
    ALL_DOCS = [dummy_doc]
    dense_retriever = vs.as_retriever(
        search_kwargs={"k": 10, "filter": {"status": {"$ne": "archived"}}}
    )
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
            for fname in existing_files:
                if allowed_file(fname):
                    ingest_documents(fname)
            print("✅ Ingestion complete!")

    except Exception as e:
        print(f"❌ Sync error: {e}")      

def archive_file_chunks(file_id):
    """
    Marks all chunks belonging to file_id as archived (status='archived')
    directly in the ChromaDB collection, via a real metadata update —
    not just an in-memory mutation. This is what makes the archive
    reliably query-time-filterable: ChromaDB reads the updated status
    the very next time it's searched.

    Returns the number of chunks archived.
    """
    global vectorstore, ALL_DOCS

    if not vectorstore:
        print("⚠️ archive_file_chunks called but vectorstore is not initialized.")
        return 0

    try:
        matches = vectorstore.get(where={"file_id": file_id}, include=["metadatas"])
    except Exception as e:
        print(f"⚠️ Could not query ChromaDB for file_id={file_id}: {e}")
        return 0

    matching_ids = matches.get("ids", [])
    matching_metadatas = matches.get("metadatas", [])

    if not matching_ids:
        print(f"⚠️ No chunks found with file_id={file_id} — nothing archived.")
        return 0

    updated_metadatas = []
    for meta in matching_metadatas:
        meta = dict(meta or {})
        meta["status"] = "archived"
        updated_metadatas.append(meta)

    try:
        vectorstore._collection.update(ids=matching_ids, metadatas=updated_metadatas)
        print(f"✅ Archived {len(matching_ids)} chunks for file_id={file_id} in ChromaDB.")
    except Exception as e:
        print(f"⚠️ Failed to update ChromaDB metadata for file_id={file_id}: {e}")
        return 0

    # Keep the in-memory ALL_DOCS list consistent too, since BM25 still
    # relies on a Python-side filter over this list (see is_not_archived).
    archived_count = 0
    for doc in ALL_DOCS:
        if doc.metadata.get("file_id") == file_id:
            doc.metadata["status"] = "archived"
            archived_count += 1

    global bm25_retriever
    try:
        active_docs = [d for d in ALL_DOCS if is_not_archived(d.metadata)]
        if active_docs:
            bm25_retriever = BM25Retriever.from_documents(active_docs, k=10)
            print(f"🔄 BM25 retriever rebuilt from {len(active_docs)} active chunks.")
    except Exception as e:
        print(f"⚠️ BM25 rebuild after archiving failed: {e}")

    return archived_count

def get_active_file_ids():
    """Returns the set of file_ids that currently have at least one
    non-archived chunk in ChromaDB. Used to filter document listings so
    stale ProcessedFile rows (files removed without going through the
    proper delete route, or leftover from CLI/manual testing) don't show
    up as available documents forever."""
    global vectorstore
    if not vectorstore:
        return set()
    try:
        raw = vectorstore.get(
            where={"status": {"$ne": "archived"}},
            include=["metadatas"]
        )
        return {
            m.get("file_id") for m in raw.get("metadatas", [])
            if m and m.get("file_id")
        }
    except Exception as e:
        print(f"⚠️ get_active_file_ids failed: {e}")
        return None  # signal failure so callers can skip filtering rather than list nothing