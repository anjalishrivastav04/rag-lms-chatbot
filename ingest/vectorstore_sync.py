"""
ingest/vectorstore_sync.py
--------------------------
ChromaDB connection + LangChain index() synchronisation + desync recovery.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document


class VectorstoreSync:
    """
    Connects to the ChromaDB HTTP server and synchronises a list of chunks
    using LangChain's incremental `index()` call.  Handles desync recovery
    (full rebuild) when the record manager gets out of step.
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._record_manager = SQLRecordManager(
            namespace="rag-chatbot",
            db_url="sqlite:///record_manager_cache.db",
        )
        self._record_manager.create_schema()

    # ── Public API ──────────────────────────────────────────────────────────

    def sync(self, chunks: list[Document]) -> dict:
        """
        Upsert *chunks* into ChromaDB using incremental deduplication.
        Returns the sync stats dict from LangChain's index().
        Falls back to a full rebuild if a desync ValueError is raised.
        """
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        print("💾 Connecting to ChromaDB server for synchronization...")

        vectorstore = Chroma(
            client=chroma_client,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )

        print(f"🔀 Running parallel deduplication for {len(chunks)} incoming chunks...")

        try:
            sync_stats = index(
                chunks,
                self._record_manager,
                vectorstore,
                cleanup="incremental",
                source_id_key="source",
            )
            self._print_summary(sync_stats)
            return sync_stats

        except ValueError as val_err:
            print(f"⚠️ Index Sync Desynchronization Detected: ({val_err})")
            print("🔄 Performing automatic index reconciliation rebuild...")
            return self._rebuild(chroma_client, chunks)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _rebuild(self, chroma_client, chunks: list[Document]) -> dict:
        """Full rebuild path — wipes the record manager and re-indexes from scratch."""
        try:
            self._record_manager.delete_keys(self._record_manager.list_keys())
            print("🗑️ Cleared record manager tracking state (was at risk of desync).")
        except Exception as rm_err:
            print(f"⚠️ Could not clear record manager cleanly: {rm_err}")

        # Delete and recreate the collection
        old_vs = Chroma(
            client=chroma_client,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        old_vs.delete_collection()

        Chroma.from_documents(
            chunks,
            self._embeddings,
            client=chroma_client,
            collection_name=self.COLLECTION_NAME,
        )
        print("✅ Vector database re-indexed from this run's chunks only.")
        print("⚠️ NOTE: This was a partial recovery — only chunks from this run are indexed.")
        print("⚠️ Files indexed in prior runs and NOT re-uploaded may be MISSING.")
        print("⚠️ Consider re-uploading all documents to be safe.")
        return {}

    @staticmethod
    def _print_summary(sync_stats: dict) -> None:
        print(f"📊 Synchronization Report: {sync_stats}")
        print(f"\n{'='*80}")
        print(f"✅ INGESTION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Added:   {sync_stats.get('num_added', 0)} new chunks")
        print(f"⏭️  Skipped: {sync_stats.get('num_skipped', 0)} existing chunks")
        print(f"🗑️  Deleted: {sync_stats.get('num_deleted', 0)} old chunks")
        print(f"📊 Updated: {sync_stats.get('num_updated', 0)} chunks")
        print(f"{'='*80}\n")
