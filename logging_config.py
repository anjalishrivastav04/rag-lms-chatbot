"""
logging_config.py
------------------
Central logging setup for the RAG chatbot system.

Design goals (why it's built this way):
1. Every log line is JSON -> easy to grep, easy to pipe into Postgres/ELK later.
2. Every request gets a `trace_id` that follows it across process boundaries:
   Flask route -> Kafka producer -> worker.py consumer -> FAISS/BM25 -> Redis -> response.
   This is the single most useful thing for debugging an async, multi-service system.
3. Log rotation so files don't grow unbounded on the server.
4. One function to call from app.py, one function to call from worker.py.

Usage:
    # In app.py (Flask process)
    from logging_config import setup_logging, RequestIdMiddleware
    logger = setup_logging("api")
    app.wsgi_app = RequestIdMiddleware(app.wsgi_app)

    # In worker.py (Kafka consumer process)
    from logging_config import setup_logging, set_trace_id
    logger = setup_logging("worker")
    # when consuming a message that carries a trace_id header:
    set_trace_id(message.headers.get("trace_id"))
    logger.info("kafka_message_consumed", extra={"topic": msg.topic})
"""

import logging
import logging.handlers
import json
import uuid
import contextvars
import os
from datetime import datetime, timezone

# ANSI colour codes (gracefully ignored on Windows / non-tty)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_WHITE  = "\033[37m"
_GREY   = "\033[90m"

# ---------------------------------------------------------------------------
# 1. Context variable that holds the current trace_id.
#    contextvars (not thread-local / not flask.g) because it works correctly
#    across both Flask's request context AND async/Kafka worker code.
# ---------------------------------------------------------------------------
_trace_id_ctx = contextvars.ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str | None = None) -> str:
    """Set (or generate) the trace_id for the current context. Returns it."""
    tid = trace_id or str(uuid.uuid4())
    _trace_id_ctx.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_ctx.get()


class TraceIdFilter(logging.Filter):
    """Injects the current trace_id into every log record automatically."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


# ---------------------------------------------------------------------------
# 2. JSON formatter — one log line = one JSON object.
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Anything passed via extra={...} gets merged in, e.g.:
        # logger.info("faiss_lookup", extra={"query_id": qid, "latency_ms": 42})
        reserved = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "trace_id", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# 2b. Pretty console formatter — human-readable, emoji-tagged lines.
#     JSON stays in the file; this goes to stdout only.
# ---------------------------------------------------------------------------
_LEVEL_ICONS = {
    "DEBUG":    f"{_GREY}🔍 DEBUG{_RESET}",
    "INFO":     f"{_GREEN}✅ INFO {_RESET}",
    "WARNING":  f"{_YELLOW}⚠️  WARN {_RESET}",
    "ERROR":    f"{_RED}❌ ERROR{_RESET}",
    "CRITICAL": f"{_RED}{_BOLD}🔥 CRIT {_RESET}",
}

_MSG_ICONS = {
    # worker lifecycle
    "worker_starting":                  "🚀",
    "worker_consumer_loop_started":      "👂",
    "worker_stopped_by_user":           "🛑",
    "worker_shutdown_complete":         "🔒",
    "vectorstore_initialized":          "📚",
    "vectorstore_reload_thread_started":"🔄",
    "vectorstore_reloaded_via_signal":  "🔄",
    "langgraph_initialized":            "🕸️ ",
    # message processing
    "message_processing_started":       "📨",
    "message_processing_completed":     "✔️ ",
    "message_processing_failed":        "💥",
    "worker_message_timeout":           "⏰",
    "worker_message_processing_error":  "💥",
    "result_saved":                     "💾",
    # answer / cache
    "langgraph_answer_complete":        "🤖",
    "question_decomposed":              "🔀",
    "low_confidence_answer":            "🎯",
    "worker_consumer_iteration_failed": "🔁",
    # retrieval
    "retrieval_completed":              "🔍",
    "chunks_selected":                  "📋",
    "retrieved_chunk":                  "📄",
    "graph_chunk_injected":             "🕸️ ",
    "rerank_completed":                 "🏆",
    "rag_fusion_completed":             "🌊",
    "cache_hit":                        "⚡",
    "blacklist_filter_applied":         "🚫",
    "graph_entities_added":             "🕸️ ",
}


class PrettyConsoleFormatter(logging.Formatter):
    """Colourised, emoji-tagged single-line formatter for terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now().strftime("%H:%M:%S")
        level_tag = _LEVEL_ICONS.get(record.levelname, record.levelname)
        msg = record.getMessage()
        icon = _MSG_ICONS.get(msg, "•")

        # Build the extra fields string (skip internal Python attrs)
        reserved = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "trace_id", "taskName",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in reserved and not k.startswith("_")
        }
        extras_str = "  " + "  ".join(
            f"{_CYAN}{k}{_RESET}={_BOLD}{v}{_RESET}" for k, v in extras.items()
        ) if extras else ""

        trace = getattr(record, "trace_id", "-")
        trace_short = trace[:8] if trace and trace != "-" else "--------"

        line = (
            f"{_DIM}[{now}]{_RESET} "
            f"{level_tag} "
            f"{_GREY}│{_RESET} "
            f"{_DIM}{record.name:<8}{_RESET} "
            f"{_GREY}│{_RESET} "
            f"{icon} {_BOLD}{msg}{_RESET}"
            f"{extras_str}"
            f"  {_GREY}trace={trace_short}{_RESET}"
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ---------------------------------------------------------------------------
# 3. setup_logging() — call once per process (once in app.py, once in worker.py)
# ---------------------------------------------------------------------------
def setup_logging(
    service_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,               # keep 5 rotated files
) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Read LOG_LEVEL env var: set LOG_LEVEL=DEBUG in .env to see per-chunk logs
    env_level_str = os.getenv("LOG_LEVEL", "").upper()
    console_level = getattr(logging, env_level_str, level)

    logger = logging.getLogger(service_name)
    logger.setLevel(min(level, console_level))  # logger must be at least as low as handlers
    logger.handlers.clear()  # avoid duplicate handlers on reload (Flask debug mode)

    formatter = JsonFormatter()
    trace_filter = TraceIdFilter()

    # Rotating file handler -> logs/api.log, logs/worker.log etc.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, f"{service_name}.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(trace_filter)
    logger.addHandler(file_handler)

    # Console handler — pretty human-readable format for the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)  # ✅ Respects LOG_LEVEL env var
    console_handler.setFormatter(PrettyConsoleFormatter())
    console_handler.addFilter(trace_filter)
    logger.addHandler(console_handler)

    # Separate error-only file — makes it trivial to tail just the failures
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, f"{service_name}.error.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(trace_filter)
    logger.addHandler(error_handler)

    return logger


# ---------------------------------------------------------------------------
# 4. Flask WSGI middleware — assigns a trace_id to every incoming request,
#    reading it from the X-Trace-Id header if the frontend already sent one
#    (e.g. carried over from a polling request), otherwise generating fresh.
# ---------------------------------------------------------------------------
class RequestIdMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        incoming = environ.get("HTTP_X_TRACE_ID")
        trace_id = set_trace_id(incoming)

        def custom_start_response(status, headers, exc_info=None):
            headers.append(("X-Trace-Id", trace_id))
            return start_response(status, headers, exc_info)

        return self.wsgi_app(environ, custom_start_response)