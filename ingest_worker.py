"""
ingest_worker.py
-----------------
Consumes document-ingestion jobs from Kafka and processes them across a
pool of real OS processes (ProcessPoolExecutor), so multiple users
uploading files at the same time actually get processed on separate CPU
cores in parallel — not queued behind each other or limited by Python's
GIL the way threads would be.

Key design point: models (embeddings, EasyOCR, etc.) are loaded ONCE per
worker process via the executor's `initializer`, not once per file. With
MAX_INGEST_WORKERS processes, that means MAX_INGEST_WORKERS x the model
load cost at startup — but every file after that reuses the already-loaded
models in whichever process picks it up, instead of reloading them per job.
"""

import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from kafka_handler import get_ingestion_consumer

logger = logging.getLogger("ingest_worker")
logging.basicConfig(level=logging.INFO)

MAX_INGEST_WORKERS = int(os.getenv("MAX_INGEST_WORKERS", "4"))

# Redis client used by the MAIN process (owns the executor) to publish
# results back for the /upload/result/<request_id> polling endpoint.
from extensions import redis_client

INGEST_RESULT_TTL = 300  # seconds


# ============================================================
# --- PER-PROCESS INITIALIZER (runs ONCE per worker process) ---
# ============================================================

def _init_worker():
    """
    Runs once when each OS process in the pool starts — NOT once per file.
    Pushes a Flask app context so DB/Chroma/Redis-dependent code inside
    ingest_documents() and save_processed_file_info() works normally,
    and lets all the heavy ML models load a single time per process.
    """
    from app import app
    ctx = app.app_context()
    ctx.push()
    print(f"✅ Ingestion worker process {os.getpid()} initialized")


# ============================================================
# --- THE ACTUAL INGESTION JOB (runs inside a worker process) ---
# ============================================================

def _run_ingestion(payload):
    """
    Executes entirely inside a pre-initialized worker process (app context
    already active via _init_worker): ingest -> reload vectorstore ->
    save DB record -> return a status dict.
    """
    file_name = payload["file_name"]
    user_id = payload["user_id"]
    filepath = payload["filepath"]

    from ingest import ingest_documents
    from routes.admin import save_processed_file_info
    from services.vectorstore import reload_vectorstore

    try:
        chunk_counts = ingest_documents(file_name)

        if not chunk_counts:
            if os.path.exists(filepath):
                os.remove(filepath)
            return {
                "success": False,
                "message": f"❌ '{file_name}' could not be processed. Only valid PDF files are supported."
            }

        reload_vectorstore()
        real_count = chunk_counts.get(file_name, 0)
        save_processed_file_info(user_id, file_name, filepath, chunk_count=real_count)

        return {
            "success": True,
            "message": f"✅ '{file_name}' uploaded and processed successfully!",
            "chunk_count": real_count
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


# ============================================================
# --- RESULT HANDLING (runs in the MAIN process) ---
# ============================================================

def _make_done_callback(request_id):
    """Fires in the main process once a worker process finishes a job.
    Writes the result to Redis for the polling endpoint to pick up."""
    def _callback(future):
        try:
            result = future.result()
        except Exception as e:
            result = {"success": False, "message": f"Worker process error: {e}"}
        try:
            redis_client.setex(f"ingest_result:{request_id}", INGEST_RESULT_TTL, json.dumps(result))
        except Exception as e:
            print(f"⚠️ Failed to save ingestion result to Redis: {e}")
        print(f"💾 Ingestion result saved for request {request_id}: success={result.get('success')}")
    return _callback


# ============================================================
# --- MAIN LOOP ---
# ============================================================

def main():
    print(f"🚀 Starting ingest_worker with {MAX_INGEST_WORKERS} parallel OS processes")
    consumer = get_ingestion_consumer(group_id="ingestion-worker-group")
    executor = ProcessPoolExecutor(max_workers=MAX_INGEST_WORKERS, initializer=_init_worker)

    try:
        for message in consumer:
            payload = message.value
            request_id = payload["request_id"]
            print(f"📥 Dispatching ingestion job {request_id} ({payload['file_name']}) to process pool")

            future = executor.submit(_run_ingestion, payload)
            future.add_done_callback(_make_done_callback(request_id))
    except KeyboardInterrupt:
        print("⏹️ Ingestion worker stopped by user")
    finally:
        executor.shutdown(wait=True)
        consumer.close()
        print("✅ Ingestion worker shutdown complete")


if __name__ == "__main__":
    main()