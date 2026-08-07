import redis
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker
import tiktoken
import os
from llm_provider import create_chat_model

# ============================================================
# --- EXTENSIONS (initialized without app, bound later) ---
# ============================================================

db = SQLAlchemy()
mail = Mail()
csrf = CSRFProtect()

# --- REDIS ---
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except Exception:
    class _UnavailableRedis:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

    redis_client = _UnavailableRedis()

# --- EMBEDDINGS ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- LLMs ---
llm = create_chat_model(temperature=0.5)
eval_llm = create_chat_model(temperature=0.0)

# --- RERANKER ---
reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="reranker_cache")

# --- TOKEN COUNTER ---
token_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(token_encoder.encode(text))