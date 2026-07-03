import redis
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from flashrank import Ranker
import tiktoken
import os

# ============================================================
# --- EXTENSIONS (initialized without app, bound later) ---
# ============================================================

db = SQLAlchemy()
mail = Mail()
csrf = CSRFProtect()

# --- REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# --- EMBEDDINGS ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- LLMs ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)
eval_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- RERANKER ---
reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="reranker_cache")

# --- TOKEN COUNTER ---
token_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(token_encoder.encode(text))