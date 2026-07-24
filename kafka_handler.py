"""
kafka_handler.py
Handles Kafka producer (sending chat requests to queue) and consumer 
(processing them one at a time through the RAG pipeline).
"""

import json
import uuid
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
CHAT_REQUESTS_TOPIC = "chat-requests"
CHAT_RESULTS_TOPIC = "chat-results"
KAFKA_BROKER = KAFKA_BOOTSTRAP_SERVERS
KAFKA_TOPIC = CHAT_REQUESTS_TOPIC
KAFKA_ENABLED = True

# --- PRODUCER (used inside Flask /chat route) ---
# ============================================================

_producer = None
INGESTION_REQUESTS_TOPIC = "document-ingestion"

# ============================================================
# --- INGESTION PRODUCER (used inside Flask /upload route) ---
# ============================================================

_ingestion_producer = None

def get_ingestion_producer():
    global _ingestion_producer
    if _ingestion_producer is None:
        try:
            _ingestion_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            print("✅ Kafka Ingestion Producer connected!")
        except Exception as e:
            print(f"⚠️ Kafka Ingestion Producer unavailable: {e}")
            _ingestion_producer = None
    return _ingestion_producer


def send_ingestion_request(user_id, filename, filepath):
    """Push a document-ingestion job onto Kafka. Returns request_id, or
    None if Kafka is unavailable (caller should fall back to direct mode)."""
    global _ingestion_producer
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "file_name": filename,
        "filepath": filepath
    }
    try:
        producer = get_ingestion_producer()
        if producer is None:
            return None
        producer.send(INGESTION_REQUESTS_TOPIC, value=payload)
        producer.flush()
        print(f"📤 Sent ingestion request to Kafka: {request_id}")
        return request_id
    except KafkaError as e:
        print(f"❌ Kafka ingestion send error: {e}")
        _ingestion_producer = None
        return None
    except Exception as e:
        print(f"❌ Unexpected Kafka ingestion error: {e}")
        _ingestion_producer = None
        return None


# ============================================================
# --- INGESTION CONSUMER (used inside ingest_worker.py) ---
# ============================================================

def get_ingestion_consumer(group_id="ingestion-worker-group"):
    """Unlike the chat consumer, this does NOT limit max_poll_records to 1 —
    we WANT to pull multiple pending jobs so they can be dispatched to
    separate OS processes and run concurrently."""
    consumer = KafkaConsumer(
        INGESTION_REQUESTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    return consumer

def get_producer():
    """Lazily create and reuse a single Kafka producer instance with auto-reconnect."""
    global _producer
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            print("✅ Kafka Producer connected!")
        except Exception as e:
            print(f"⚠️ Kafka Producer unavailable: {e}")
            _producer = None
    return _producer


def send_chat_request(user_id, session_id, user_message):
    """
    Push a chat request onto the Kafka topic.
    Returns request_id if successful, None if Kafka unavailable (fallback to direct mode).
    """
    global _producer
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "message": user_message
    }
    try:
        producer = get_producer()
        if producer is None:
            print("⚠️ Kafka unavailable — will use direct mode")
            return None
        producer.send(CHAT_REQUESTS_TOPIC, value=payload)
        producer.flush()
        print(f"📤 Sent request to Kafka: {request_id}")
        return request_id
    except KafkaError as e:
        print(f"❌ Kafka send error: {e}")
        # ✅ Reset producer so next request tries to reconnect
        _producer = None
        return None
    except Exception as e:
        print(f"❌ Unexpected Kafka error: {e}")
        _producer = None
        return None
# ============================================================
# --- CONSUMER (used inside the separate worker script) ---
# ============================================================

def get_consumer(group_id="rag-worker-group"):
    """Create a Kafka consumer that reads chat requests one at a time."""
    consumer = KafkaConsumer(
        CHAT_REQUESTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        max_poll_records=1  # ✅ Process ONE message at a time (per requirement)
    )
    return consumer


# ============================================================
# --- VECTORSTORE UPDATE NOTIFICATIONS ---
# (Lets ingest_worker.py / ingest.py tell the chat worker.py process
#  "new data is available" without them sharing memory. Previously,
#  worker.py only ever read ChromaDB's contents ONCE at startup, so any
#  ingestion happening in a different process — CLI or the production
#  ingest_worker.py pool — never became visible to live chat answers
#  until worker.py was manually restarted. This closes that gap.)
# ============================================================

VECTORSTORE_UPDATE_TOPIC = "vectorstore-updates"

_vectorstore_update_producer = None

def get_vectorstore_update_producer():
    global _vectorstore_update_producer
    if _vectorstore_update_producer is None:
        try:
            _vectorstore_update_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            print("✅ Kafka Vectorstore-Update Producer connected!")
        except Exception as e:
            print(f"⚠️ Kafka Vectorstore-Update Producer unavailable: {e}")
            _vectorstore_update_producer = None
    return _vectorstore_update_producer


def send_vectorstore_update(filename):
    """Notify any listening process (chat worker.py) that ChromaDB has
    new/changed content, so it can reload its retriever. Best-effort —
    failure here should never block ingestion itself."""
    global _vectorstore_update_producer
    try:
        producer = get_vectorstore_update_producer()
        if producer is None:
            return False
        producer.send(VECTORSTORE_UPDATE_TOPIC, value={"filename": filename})
        producer.flush()
        print(f"📢 Sent vectorstore-update notification for: {filename}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send vectorstore-update notification: {e}")
        _vectorstore_update_producer = None
        return False


def get_vectorstore_update_consumer(group_id="vectorstore-update-listeners"):
    """Each worker.py instance should use its OWN unique group_id (e.g.
    including its process id) so ALL running chat workers get every
    update — unlike chat-requests, this is a broadcast, not a queue."""
    consumer = KafkaConsumer(
        VECTORSTORE_UPDATE_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",  # only care about updates from now on
        enable_auto_commit=True,
    )
    return consumer