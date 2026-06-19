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