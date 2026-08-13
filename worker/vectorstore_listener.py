"""
worker/vectorstore_listener.py
------------------------------
Background daemon thread that listens for Redis pubsub signals
and triggers a vectorstore reload when a new document is uploaded.
"""

from extensions import redis_client
from services.vectorstore import initialize_vectorstore


def listen_for_vectorstore_updates(app, logger) -> None:
    """
    Subscribe to the 'vectorstore_updates' Redis channel and reload
    the vectorstore whenever a message arrives.

    Designed to run inside a daemon thread — pass the Flask *app*
    and a configured *logger* at startup.
    """
    pubsub = redis_client.pubsub()
    pubsub.subscribe("vectorstore_updates")
    logger.info("vectorstore_reload_listener_started")

    for message in pubsub.listen():
        if message["type"] == "message":
            with app.app_context():
                initialize_vectorstore()
                logger.info("vectorstore_reloaded_via_signal")
