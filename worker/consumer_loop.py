"""
worker/consumer_loop.py
-----------------------
Kafka consumer poll loop, ThreadPoolExecutor timeout guard, and
the worker startup sequence (vectorstore init, reload thread, main).
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app import app
from kafka_handler import get_consumer
from logging_config import setup_logging
from services.vectorstore import initialize_vectorstore
from worker.vectorstore_listener import listen_for_vectorstore_updates
from worker.message_processor import process_message
from worker.result_store import timeout_result, REDIS_RESULT_TTL

logger = setup_logging("worker")

USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
WORKER_TIMEOUT_SECONDS = 90  # kill a hung LLM call after 90 s

_executor = ThreadPoolExecutor(max_workers=1)

# ── LangGraph (optional) ──────────────────────────────────────────────────────
if USE_LANGGRAPH:
    from graph_answer import build_graph
    _graph = None  # built lazily on first use

    def get_graph():
        global _graph
        if _graph is None:
            _graph = build_graph()
            logger.info("langgraph_initialized")
        return _graph


# ── Consumer poll loop ────────────────────────────────────────────────────────

def _run_consumer(consumer):
    """Inner loop — extracted so main() stays readable."""
    logger.info("worker_consumer_loop_started")
    while True:
        try:
            # poll() returns after timeout_ms even with no messages so that
            # Kafka's background heartbeat sender is never starved.
            records = consumer.poll(timeout_ms=5000)
            for _tp, messages in records.items():
                for message in messages:
                    payload = message.value
                    request_id = payload.get("request_id", "unknown")
                    future = _executor.submit(process_message, payload, logger)
                    try:
                        future.result(timeout=WORKER_TIMEOUT_SECONDS)
                    except FuturesTimeoutError:
                        logger.error("worker_message_timeout", extra={
                            "request_id": request_id,
                            "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                        })
                        timeout_result(request_id, WORKER_TIMEOUT_SECONDS)
                    except Exception:
                        logger.error("worker_message_processing_error",
                                     exc_info=True, extra={"request_id": request_id})
        except Exception as exc:
            logger.warning("worker_consumer_iteration_failed",
                           exc_info=True, extra={"error": str(exc)})
            time.sleep(5)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    from llm_provider import get_llm_provider

    logger.info("worker_starting", extra={
        "topic": "chat-requests",
        "llm_provider": get_llm_provider(),
        "using_langgraph": USE_LANGGRAPH,
        "timeout_s": WORKER_TIMEOUT_SECONDS,
    })

    # Initialise vectorstore inside app context
    with app.app_context():
        initialize_vectorstore()
        logger.info("vectorstore_initialized")

    # Background reload thread
    reload_thread = threading.Thread(
        target=listen_for_vectorstore_updates,
        args=(app, logger),
        daemon=True,
    )
    reload_thread.start()
    logger.info("vectorstore_reload_thread_started")

    consumer = get_consumer(group_id="rag-worker-group")

    try:
        _run_consumer(consumer)
    except KeyboardInterrupt:
        logger.info("worker_stopped_by_user")
    finally:
        consumer.close()
        logger.info("worker_shutdown_complete")
