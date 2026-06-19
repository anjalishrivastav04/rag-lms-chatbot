import os
import uuid
import time
import json
import redis
import shutil
import hashlib
import re
import numpy as np
import threading
from flask import Flask, request, jsonify, render_template, redirect
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from flask import send_from_directory, abort
from langchain_core.documents import Document
from ocr_handler import save_ocr_text_to_file, extract_text_from_image
from flashrank import Ranker, RerankRequest
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
from ingest import ingest_documents
from graph_handler import graph_retrieve, delete_graph_for_file
from kafka_handler import send_chat_request, KAFKA_BROKER, KAFKA_TOPIC, KAFKA_ENABLED
from dotenv import load_dotenv
from sqlalchemy import func

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
app = Flask(__name__)

# ✅ Initialize embedder for semantic cache
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedder loaded successfully!")

app.secret_key = "123456"

# --- FILE UPLOAD CONFIG ---
UPLOAD_FOLDER = "documents"
ALLOWED_EXTENSIONS = {"pdf", "txt", "jpg", "jpeg", "png", "bmp", "gif"}
OCR_SUPPORTED = {"jpg", "jpeg", "png", "bmp", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- POSTGRESQL CONFIG ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/postgres'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 20,
    'max_overflow': 40,
    'pool_recycle': 1800,
    'pool_pre_ping': True,
    'pool_timeout': 30
}
db = SQLAlchemy(app)

# --- REDIS CONFIG ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDIS_TTL = 3600
RATE_LIMIT_MAX = 200        # max messages
RATE_LIMIT_WINDOW = 4 * 60 * 60  # 4 hours in seconds
# ============================================================
# --- DATABASE MODELS ---
# ============================================================

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    cache_source = db.Column(db.String(50), default='NONE')
    response_time_ms = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "cache_source": self.cache_source,
            "response_time_ms": self.response_time_ms,
            "created_at": str(self.created_at)
        }

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    chats = db.relationship('ChatHistory', backref='user', lazy=True,
                            foreign_keys='ChatHistory.user_id',
                            cascade='all, delete-orphan')

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "created_at": str(self.created_at)
        }

class RateLimit(db.Model):
    __tablename__ = 'rate_limits'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message_count = db.Column(db.Integer, default=0)
    window_start = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'message_count': self.message_count,
            'window_start': str(self.window_start)
        }
    
class ProcessedFile(db.Model):
    __tablename__ = 'processed_files'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.String(36), unique=True, nullable=False, 
                        default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    file_size = db.Column(db.Integer)
    processed_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    chunk_count = db.Column(db.Integer, default=0)
    version = db.Column(db.String(50))

    def to_dict(self):
        return {
            "filename": self.filename,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "processed_at": str(self.processed_at),
            "chunk_count": self.chunk_count,
            "version": self.version
        }

class SemanticCacheRecord(db.Model):
    __tablename__ = 'semantic_cache'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    query_text = db.Column(db.Text, nullable=False)
    query_embedding = db.Column(db.Text)
    response = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(50), default='general')
    hit_count = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)

    def is_expired(self):
        return datetime.utcnow() > self.expires_at

    def time_remaining(self):
        if self.is_expired():
            return "Expired"
        delta = self.expires_at - datetime.utcnow()
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f} hours"

    def to_dict(self):
        return {
            'id': self.id,
            'query': self.query_text,
            'response': self.response,
            'content_type': self.content_type,
            'hits': self.hit_count,
            'created': str(self.created_at),
            'expires': str(self.expires_at),
            'ttl_remaining': self.time_remaining()
        }

class ResponseEvaluation(db.Model):
    __tablename__ = 'response_evaluations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(255), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text)
    score = db.Column(db.Integer)
    feedback = db.Column(db.Text)
    evaluated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'score': self.score,
            'feedback': self.feedback,
            'evaluated_at': str(self.evaluated_at)
        }


class AdminFeedback(db.Model):
    __tablename__ = 'admin_feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    original_answer = db.Column(db.Text, nullable=False)
    feedback_type = db.Column(db.String(10), nullable=False)
    correct_answer = db.Column(db.Text)
    is_ingested = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'original_answer': self.original_answer,
            'feedback_type': self.feedback_type,
            'correct_answer': self.correct_answer,
            'is_ingested': self.is_ingested,
            'created_at': str(self.created_at)
        }

# ============================================================
# --- ADMIN REQUIRED DECORATOR ---
# ============================================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                "success": False,
                "message": "Please login first!",
                "error": "not_authenticated"
            }), 401
        user = User.query.get(user_id)
        if not user:
            session.clear()
            return jsonify({
                "success": False,
                "message": "User not found!"
            }), 401
        if not user.is_admin:
            return jsonify({
                "success": False,
                "message": "⛔ Admin access required!",
                "error": "not_admin"
            }), 403
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# --- TTL CONFIGURATION ---
# ============================================================

CONTENT_TTL = {
    'lecture_notes': 24,
    'assignments': 7 * 24,
    'grades': 2,
    'syllabus': 90 * 24,
    'announcements': 12,
    'course_handbook': 60 * 24,
    'admin_feedback': 90 * 24,
}
DEFAULT_TTL = 24

# ============================================================
# --- CACHE FUNCTIONS ---
# ============================================================

def cache_response_with_ttl(user_id, query_text, query_embedding, response, content_type='general'):
    try:
        ttl_hours = CONTENT_TTL.get(content_type, DEFAULT_TTL)
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        embedding_str = json.dumps(query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding)
        cache_record = SemanticCacheRecord(
            user_id=user_id,
            query_text=query_text,
            query_embedding=embedding_str,
            response=response,
            content_type=content_type,
            expires_at=expires_at
        )
        db.session.add(cache_record)
        db.session.commit()
        print(f"✅ Cached '{content_type}' for {ttl_hours} hours")
        return expires_at
    except Exception as e:
        print(f"❌ Cache store error: {e}")
        db.session.rollback()
        return None

def get_cached_response(user_id, query_embedding, threshold=0.85):
    try:
        cached_records = SemanticCacheRecord.query.filter(
            SemanticCacheRecord.user_id == user_id,
            SemanticCacheRecord.expires_at > datetime.utcnow()
        ).all()
        if not cached_records:
            return None, None, 0
        best_match = None
        best_similarity = 0
        for record in cached_records:
            cached_embedding = np.array(json.loads(record.query_embedding))
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = record
        if best_match:
            best_match.hit_count += 1
            db.session.commit()
            return best_match.response, best_match.content_type, best_similarity
        return None, None, 0
    except Exception as e:
        print(f"❌ Cache retrieval error: {e}")
        return None, None, 0

def cleanup_expired_cache():
    try:
        expired_count = SemanticCacheRecord.query.filter(
            SemanticCacheRecord.expires_at < datetime.utcnow()
        ).delete()
        db.session.commit()
        if expired_count > 0:
            print(f"🗑️ Cleaned up {expired_count} expired cache entries")
        return expired_count
    except Exception as e:
        print(f"❌ Cache cleanup error: {e}")
        db.session.rollback()
        return 0

# ============================================================
# --- DB HELPERS ---
# ============================================================

def save_chat_message(user_id, session_id, role, content, cache_source='NONE', response_time_ms=0):
    try:
        msg = ChatHistory(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            cache_source=cache_source,
            response_time_ms=response_time_ms
        )
        db.session.add(msg)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"❌ DB Error: {e}")

def get_past_chat_history(session_id, limit=6):
    try:
        history = ChatHistory.query.filter_by(session_id=session_id)\
                                   .order_by(ChatHistory.created_at.desc())\
                                   .limit(limit).all()
        history.reverse()
        formatted = ""
        for msg in history:
            label = "Student" if msg.role == "user" else "Assistant"
            formatted += f"{label}: {msg.content}\n"
        return formatted if formatted else "No previous history.\n"
    except Exception as e:
        print(f"❌ History fetch error: {e}")
        return "No previous history.\n"

# ============================================================
# --- EMBEDDINGS & VECTOR STORE ---
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_vectorstore():
    if os.path.exists("vectorstore") and os.path.exists(os.path.join("vectorstore", "index.faiss")):
        try:
            vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            docs = [vs.docstore.search(idx) for idx in vs.index_to_docstore_id.values()]
            valid_docs = [d for d in docs if d.metadata.get("source") != "system"]
            if valid_docs:
                print(f"📚 System successfully mapped {len(valid_docs)} document pieces into the runtime index context.")
                return vs, valid_docs
        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}. Reinitializing.")
    print("⚠️ No valid document data discovered yet. Standing by for upload files...")
    dummy_doc = Document(page_content="No documents have been uploaded yet.",
                         metadata={"source": "system", "filetype": "txt", "chunk_index": 0})
    vs = FAISS.from_documents([dummy_doc], embeddings)
    return vs, [dummy_doc]

vectorstore, ALL_DOCS = initialize_vectorstore()

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=10)
RRF_K = 60

def reciprocal_rank_fusion(bm25_docs, dense_docs):
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = doc
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = doc
    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in ranked_keys]

def hybrid_retrieve(question):
    bm25_docs = bm25_retriever.invoke(question)
    dense_docs = dense_retriever.invoke(question)
    return reciprocal_rank_fusion(bm25_docs, dense_docs)

reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="reranker_cache")

def rerank_documents(question, docs, top_n=3):
    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
        return docs[:top_n]
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = reranker.rerank(rerank_request)
    top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    top_indices = [r["id"] for r in top_results]
    return [docs[i] for i in top_indices]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))
eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# --- REDIS CACHE HELPERS ---
# ============================================================

def check_redis_cache(question):
    try:
        cache_key = f"rag:{question.lower().strip()}"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"⚠️ Redis read error: {e}")
    return None
# --- REQUEST LOCK (prevents user from spamming while one request is pending) ---
def is_user_request_pending(user_id):
    """Check if this user already has a pending request in the queue"""
    lock_key = f"pending_request:{user_id}"
    return redis_client.exists(lock_key)

def set_user_request_pending(user_id, request_id, ttl=120):
    """Mark this user as having a pending request (auto-expires after ttl seconds as safety net)"""
    lock_key = f"pending_request:{user_id}"
    redis_client.setex(lock_key, ttl, request_id)

def clear_user_request_pending(user_id):
    """Remove the pending lock once the request is processed"""
    lock_key = f"pending_request:{user_id}"
    redis_client.delete(lock_key)

def save_to_redis_cache(question, response_text):
    try:
        cache_key = f"rag:{question.lower().strip()}"
        redis_client.setex(cache_key, REDIS_TTL, json.dumps(response_text))
    except Exception as e:
        print(f"⚠️ Redis write error: {e}")

# --- SOURCE INDEX HELPERS ---
def save_source_index(filename, chunk_ids):
    """Store filename → chunk_ids mapping in Redis for O(1) deletion"""
    try:
        base_name = filename.rsplit('.', 1)[0].lower()
        for key in [filename, f"{base_name}_ocr.txt", f"{base_name}_vision.txt"]:
            if chunk_ids:
                redis_client.sadd(f"source_index:{key}", *chunk_ids)
                redis_client.expire(f"source_index:{key}", 86400 * 30)  # 30 days
        print(f"✅ Source index saved for: {filename} ({len(chunk_ids)} chunks)")
    except Exception as e:
        print(f"⚠️ Source index save error: {e}")

def get_chunk_ids_for_file(filename):
    """O(1) Redis lookup to get all chunk IDs for a file"""
    try:
        base_name = filename.rsplit('.', 1)[0].lower()
        all_ids = set()
        for key in [filename, f"{base_name}_ocr.txt", f"{base_name}_vision.txt"]:
            ids = redis_client.smembers(f"source_index:{key}")
            all_ids.update(ids)
        return list(all_ids)
    except Exception as e:
        print(f"⚠️ Source index lookup error: {e}")
        return []

def delete_source_index(filename):
    """Remove source index from Redis after file deletion"""
    try:
        base_name = filename.rsplit('.', 1)[0].lower()
        for key in [filename, f"{base_name}_ocr.txt", f"{base_name}_vision.txt"]:
            redis_client.delete(f"source_index:{key}")
        print(f"✅ Source index deleted for: {filename}")
    except Exception as e:
        print(f"⚠️ Source index delete error: {e}")

# --- DELETED FILES BLACKLIST ---
def add_to_blacklist(filename):
    """Add deleted filename to Redis blacklist"""
    try:
        base_name = filename.rsplit('.', 1)[0].lower()
        redis_client.sadd("deleted_files_blacklist", filename,
                         f"{base_name}_ocr.txt",
                         f"{base_name}_vision.txt",
                         f"{base_name}.txt")
        print(f"✅ Added {filename} to blacklist")
    except Exception as e:
        print(f"⚠️ Blacklist error: {e}")

def get_blacklist():
    """Get all deleted filenames"""
    try:
        return redis_client.smembers("deleted_files_blacklist")
    except:
        return set()

RATE_LIMIT_MAX = 200        # max messages
RATE_LIMIT_WINDOW = 4 * 60 * 60  # 4 hours in seconds

    
def check_rate_limit(user_id):
    """
    Returns (allowed, remaining, reset_in) using Redis with a 4-hour TTL.
    Key auto-expires, so no manual cleanup needed.
    """
    try:
        key = f"rate_limit:{user_id}"
        count = redis_client.incr(key)
        if count == 1:
            redis_client.expire(key, RATE_LIMIT_WINDOW)
        ttl = redis_client.ttl(key)
        reset_in = ttl if ttl > 0 else RATE_LIMIT_WINDOW
        if count > RATE_LIMIT_MAX:
            return False, 0, reset_in
        return True, RATE_LIMIT_MAX - count, reset_in
    except Exception as e:
        print(f"⚠️ Rate limit error: {e}")
        return True, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW

# ============================================================
# --- SEMANTIC CACHE (FAISS) ---
# ============================================================

CACHE_DIR = "semantic_cache_store"
DISTANCE_THRESHOLD = 0.2

def check_semantic_cache(question):
    if not os.path.exists(CACHE_DIR):
        return None
    try:
        cache_store = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
        results = cache_store.similarity_search_with_score(question, k=1)
        if results:
            cached_doc, distance = results[0]
            if distance <= DISTANCE_THRESHOLD:
                return cached_doc.metadata.get("response")
    except Exception as e:
        print(f"⚠️ Semantic cache read error: {e}")
    return None

def save_to_semantic_cache(question, response_text):
    new_doc = Document(page_content=question, metadata={"response": response_text})
    try:
        if os.path.exists(CACHE_DIR):
            cache_store = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
            cache_store.add_documents([new_doc])
            cache_store.save_local(CACHE_DIR)
        else:
            cache_store = FAISS.from_documents([new_doc], embeddings)
            cache_store.save_local(CACHE_DIR)
    except Exception as e:
        print(f"⚠️ Semantic cache write error: {e}")

# ============================================================
# --- FILE HELPERS ---
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reload_vectorstore():
    global vectorstore, dense_retriever, bm25_retriever, ALL_DOCS
    vectorstore, ALL_DOCS = initialize_vectorstore()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
    print(f"🔄 Vectorstore configuration reloaded globally.")

def is_casual_query(question):
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|wassup|yo|who are you|what is your name|how are you|what can you do|help|thanks|thank you|okay|ok|cool|nice|great|bye|goodbye)\b"
    if re.search(greetings, question.lower().strip()):
        return True
    return False

# ============================================================
# --- INGEST FEEDBACK TO VECTORSTORE ---
# ============================================================

def ingest_feedback_to_vectorstore(question, correct_answer, feedback_id):
    try:
        content = f"Q: {question}\nA: {correct_answer}"
        doc = Document(
            page_content=content,
            metadata={
                "source": f"admin_feedback_{feedback_id}",
                "filetype": "feedback",
                "chunk_index": 0,
                "question": question,
                "is_admin_feedback": True
            }
        )
        global vectorstore, dense_retriever, bm25_retriever, ALL_DOCS
        if vectorstore:
            vectorstore.add_documents([doc])
            vectorstore.save_local("vectorstore")
            ALL_DOCS.append(doc)
            bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
            cache_response_with_ttl(
                user_id=1,
                query_text=question,
                query_embedding=embedder.encode(question),
                response=correct_answer,
                content_type='admin_feedback'
            )
            print(f"✅ Feedback ingested to vectorstore: {question[:50]}")
            return True
        else:
            vectorstore = FAISS.from_documents([doc], embeddings)
            vectorstore.save_local("vectorstore")
            ALL_DOCS.append(doc)
            bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
            print(f"✅ New vectorstore created with feedback!")
            return True
    except Exception as e:
        print(f"❌ Feedback ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# --- MAIN ANSWER FUNCTION ---
# ============================================================
def is_list_documents_query(question):
    """Detect if user is asking to list available documents/files"""
    patterns = r"\b(list|show|what|which).*(document|file|pdf)s?\b"
    return bool(re.search(patterns, question.lower().strip()))


def get_answer(question, session_id):
    # ✅ FIX 1: Define past_history and blacklist BEFORE the casual query branch
    past_history = get_past_chat_history(session_id, limit=6)
    blacklist = get_blacklist()
    blacklist_str = ", ".join(blacklist) if blacklist else "None"

    if is_casual_query(question):
        print("💬 Casual query detected — routing directly to LLM conversational mode.")

        prompt = f"""You are a friendly, warm and enthusiastic document assistant named RagBot! 🤖
You love helping students and users find information from their documents.
When someone greets you, respond in a fun, warm and welcoming way.
Introduce yourself briefly and let them know what you can help with.
Keep it short, friendly and use emojis naturally.
Do NOT reference or fabricate any document content in casual conversation.
Deleted files: {blacklist_str}

### PAST CONVERSATION:
{past_history}

### QUESTION:
{question}

Answer:"""

        response = llm.invoke(prompt)
        return response.content, "NONE", "⚡ Direct LLM Conversation (No Document Search)"

    # ✅ NEW: Direct DB lookup for "list documents" queries - bypasses LLM hallucination
    if is_list_documents_query(question):
        try:
            blacklist_filenames = set(get_blacklist())
            files = ProcessedFile.query.all()
            valid_files = [f for f in files if f.filename not in blacklist_filenames]

            if not valid_files:
                return "There are no documents available right now.", "NONE", "📋 Direct database lookup"

            file_list = "\n".join([f"- {f.file_id}" for f in valid_files])
            answer = f"Here are the available documents:\n\n{file_list}"
            return answer, "NONE", "📋 Direct database lookup (no LLM hallucination)"
        except Exception as e:
            print(f"⚠️ List documents error: {e}")
            # Falls through to normal RAG pipeline if this fails

    cached = check_redis_cache(question)
    if cached:
        return cached, "REDIS", ""

    cached = check_semantic_cache(question)
    if cached:
        save_to_redis_cache(question, cached)
        return cached, "SEMANTIC", ""

    docs = hybrid_retrieve(question)
    docs = rerank_documents(question, docs, top_n=6)

    # ✅ FIX 2: Hard blacklist filter — remove any docs from deleted files
    # This ensures even stale FAISS chunks are blocked before building context
    if blacklist:
        original_count = len(docs)
        docs = [
            doc for doc in docs
            if not any(
                bl.lower().rsplit('.', 1)[0] in doc.metadata.get("source", "").lower()
                for bl in blacklist
            )
        ]
        filtered_count = original_count - len(docs)
        if filtered_count > 0:
            print(f"🚫 Blacklist filter removed {filtered_count} chunks from deleted files")

    graph_context, related_entities = graph_retrieve(question)
    extra_docs = []
    if related_entities:
        for entity in related_entities[:5]:
            entity_docs = dense_retriever.invoke(entity)
            # ✅ Also filter graph-retrieved docs through blacklist
            entity_docs = [
                d for d in entity_docs
                if not any(
                    bl.lower().rsplit('.', 1)[0] in d.metadata.get("source", "").lower()
                    for bl in blacklist
                )
            ]
            extra_docs.extend(entity_docs)
        existing_contents = {doc.page_content for doc in docs}
        for doc in extra_docs:
            if doc.page_content not in existing_contents:
                docs.append(doc)
                existing_contents.add(doc.page_content)
        print(f"🔗 Added {len(extra_docs)} extra chunks via graph entity search")

    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
        return "I don't have any uploaded documents to extract information from right now.", "NONE", ""

    context = ""
    sources_set = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filetype = doc.metadata.get("filetype", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        sources_set.add(source)

        # ✅ Look up real file_id from database
        file_record = ProcessedFile.query.filter_by(filename=source).first()
        file_id = file_record.file_id if file_record else "UNKNOWN"

        context += f"[FileID: {file_id} | Type: {filetype} | Chunk: {chunk_index}]\n"
        context += doc.page_content + "\n\n"

    # ✅ Built once, AFTER the loop — context is now fully assembled
    alpha = 0.7
    retrieval_info = f"📊 Retrieved using: {int(alpha*100)}% Semantic (FAISS) + {int((1-alpha)*100)}% Keyword (BM25)\n📎 Sources: {', '.join(sources_set)}"

    prompt = f"""You are an expert, smart, helpful document assistant. Your job is to answer questions based strictly on the provided context.

STRICT RULES — follow these without exception:
1. Answer in clean, natural, conversational language only.
2. NEVER display source tags like [Source:...], [Type:...], or [Chunk:...] in your answer.
3. NEVER reveal actual filenames to the user — always refer to documents by their File ID only.
4. When asked what files/documents are available, list only File IDs that appear EXACTLY as written in the context below. NEVER invent, guess, or generate File IDs that are not literally present in the CONTEXT section.
5. If the answer is found in the context, answer confidently and completely.
6. If multiple documents have relevant info, combine and summarize from ALL of them.
7. If the answer is NOT in the context, say clearly: "I don't have information about this in the uploaded documents."
8. NEVER make up or guess information that is not in the context.
9. If the document mentions a year or date, use that — do NOT use today's date or assume the current year is 2026.
10. The following files have been DELETED — completely IGNORE any information from them: {blacklist_str}
11. Keep your answer focused and concise — no unnecessary filler or repetition.

### PAST CONVERSATION:
{past_history}

### KNOWLEDGE GRAPH:
{graph_context}

### CONTEXT:
{context}

### QUESTION:
{question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content
    # Don't cache bad responses containing raw metadata
    if "[Source:" not in answer and "[Type:" not in answer and "[FileID:" not in answer:
        save_to_redis_cache(question, answer)
        save_to_semantic_cache(question, answer)
    else:
        print("⚠️ Bad response detected — skipping cache!")

    return answer, "NONE", retrieval_info
# ============================================================
# --- FILE HASH & PROCESSING HELPERS ---
# ============================================================

def calculate_file_hash(filepath):
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_file_version(filename):
    match = re.search(r'(_v\d+|_updated|_v\d+_updated)', filename.lower())
    return match.group(1) if match else "v1"

def sync_existing_documents():
    """Scan documents folder and add any files not in DB"""
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
            print("🔄 Running ingestion on synced documents...")
            ingest_documents()
            print("✅ Ingestion complete!")

    except Exception as e:
        print(f"❌ Sync error: {e}")

def save_processed_file_info(user_id, filename, filepath, chunk_count):
    try:
        file_hash = calculate_file_hash(filepath)
        file_size = os.path.getsize(filepath)
        version = get_file_version(filename)
        existing = ProcessedFile.query.filter_by(user_id=user_id, filename=filename).first()
        if existing:
            existing.file_hash = file_hash
            existing.file_size = file_size
            existing.chunk_count = chunk_count
            existing.version = version
            existing.processed_at = db.func.now()
        else:
            processed = ProcessedFile(
                user_id=user_id,
                filename=filename,
                file_hash=file_hash,
                file_size=file_size,
                chunk_count=chunk_count,
                version=version
            )
            db.session.add(processed)
        db.session.commit()
        print(f"💾 Saved file info for user {user_id}: {filename}")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error saving file info: {e}")

# ============================================================
# --- RAG RESPONSE EVALUATOR ---
# ============================================================

def evaluate_response(question, answer, context, user_id, session_id):
    try:
        eval_prompt = f"""You are a RAG system evaluator. Evaluate the answer fairly and practically.

QUESTION: {question}

RETRIEVED CONTEXT: {context[:1000]}

GENERATED ANSWER: {answer}

Evaluation criteria:
- RELEVANCE: Does the answer address the question asked?
- FAITHFULNESS: Is the answer based on the context (summarizing is fine, no need to copy word for word)?
- COMPLETENESS: Does the answer provide enough useful information?
- CLARITY: Is the answer clean and easy to understand?

IMPORTANT RULES:
- If the answer summarizes context in its own words, that is GOOD not bad
- If the answer combines info from multiple chunks, that is GOOD
- Only penalize if the answer makes up facts NOT in the context
- Only penalize if the answer completely ignores the question

Rate strictly on scale 1-5:
1 = Completely wrong or made up
2 = Partially relevant but missing key info
3 = Relevant and mostly correct
4 = Good answer, faithful to context
5 = Perfect answer, complete and accurate

Reply in this EXACT format only:
SCORE: <number 1-5>
FEEDBACK: <one sentence explanation>

Nothing else."""
        eval_response = eval_llm.invoke(eval_prompt)
        content = eval_response.content.strip()
        lines = content.split('\n')
        score = 3
        feedback = "Evaluation completed"
        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = int(line.replace('SCORE:', '').strip())
                    score = max(1, min(5, score))
                except:
                    score = 3
            elif line.startswith('FEEDBACK:'):
                feedback = line.replace('FEEDBACK:', '').strip()
        evaluation = ResponseEvaluation(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer,
            context=context[:2000],
            score=score,
            feedback=feedback
        )
        db.session.add(evaluation)
        db.session.commit()
        print(f"✅ Response evaluated — Score: {score}/5 | {feedback}")
        return score, feedback
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        db.session.rollback()
        return None, None

# ============================================================
# --- ROUTES ---
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"reply": "Please login first!", "error": "not_authenticated"})
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"reply": "User not found!", "error": "user_not_found"})

    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    client_session = data.get("session_id", "").strip()
    if not client_session or client_session == "default_session":
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"session-{user_id}"))
    else:
        session_id = client_session

    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    # ✅ Check pending FIRST — don't burn quota on a request that gets bounced
    if is_user_request_pending(user_id):
        return jsonify({
            "reply": "⏳ Please wait — your previous request is still being processed.",
            "error": "request_pending"
        }), 429

    # ✅ Single rate-limit check
    allowed, remaining, reset_in = check_rate_limit(user_id)
    if not allowed:
        hours = reset_in // 3600
        mins = (reset_in % 3600) // 60
        return jsonify({
            "reply": f"⛔ You have reached the limit of {RATE_LIMIT_MAX} messages per 4-hour session. Please try again in {hours}h {mins}m.",
            "error": "rate_limit_exceeded",
            "reset_in": reset_in
        })
    
    try:
        start_time = time.time()
        save_chat_message(user_id, session_id, "user", user_message)

        # ✅ TIER 0: Redis exact match (fast path - skip Kafka entirely if cached!)
        exact_cache_key = f"user:{user_id}:exact:{hashlib.md5(user_message.encode()).hexdigest()}"
        redis_cached = redis_client.get(exact_cache_key)
        if redis_cached:
            duration = int((time.time() - start_time) * 1000)
            cached_data = json.loads(redis_cached)
            save_chat_message(user_id, session_id, "assistant", cached_data['reply'],
                              cache_source="REDIS_EXACT", response_time_ms=duration)
            def run_eval_redis():
                with app.app_context():
                    evaluate_response(user_message, cached_data['reply'],
                                      cached_data.get('retrieval_info', ''), user_id, session_id)
            threading.Thread(target=run_eval_redis, daemon=True).start()
            return jsonify({
                "reply": cached_data['reply'],
                "retrieval_info": cached_data.get('retrieval_info', ''),
                "cache_source": "REDIS_EXACT ⚡",
                "duration_ms": duration,
                "user": user.username,
                "done": True
            })

        # ✅ TIER 1: Semantic Cache (fast path - skip Kafka if cached!)
        query_embedding = embedder.encode(user_message)
        cached_response, content_type, similarity = get_cached_response(
            user_id, query_embedding, threshold=0.85)
        if cached_response:
            duration = int((time.time() - start_time) * 1000)
            save_chat_message(user_id, session_id, "assistant", cached_response,
                              cache_source="SEMANTIC_CACHE", response_time_ms=duration)
            def run_eval_semantic():
                with app.app_context():
                    evaluate_response(user_message, cached_response,
                                      f"Similarity: {similarity:.2%}", user_id, session_id)
            threading.Thread(target=run_eval_semantic, daemon=True).start()
            return jsonify({
                "reply": cached_response,
                "retrieval_info": f"Similarity: {similarity:.2%}",
                "cache_source": f"SEMANTIC_CACHE ({content_type}) 🧠",
                "duration_ms": duration,
                "user": user.username,
                "done": True
            })

        # ✅ TIER 2: Not cached - push to Kafka queue instead of processing directly!
        request_id = send_chat_request(user_id, session_id, user_message)
        if not request_id:
            return jsonify({"reply": "❌ Failed to queue your request. Please try again.", "done": True})

        set_user_request_pending(user_id, request_id)

        return jsonify({
            "reply": "⏳ Processing your request...",
            "request_id": request_id,
            "done": False
        })
    except Exception as e:
        clear_user_request_pending(user_id)
        return jsonify({"reply": f"Error: {str(e)}", "done": True})


@app.route("/chat/result/<request_id>", methods=["GET"])
def chat_result(request_id):
    """Frontend polls this to check if the Kafka-processed answer is ready"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"done": False, "error": "not_authenticated"})

    result_key = f"chat_result:{request_id}"
    result = redis_client.get(result_key)

    if result:
        data = json.loads(result)
        redis_client.delete(result_key)  # Clean up after delivering
        clear_user_request_pending(user_id)  # ✅ Unlock user for next request
        return jsonify({**data, "done": True})

    return jsonify({"done": False})

@app.route("/rate-limit/status", methods=["GET"])
def rate_limit_status():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "not_authenticated"}), 401
    try:
        key = f"rate_limit:{user_id}"
        count = redis_client.get(key)
        if count is None:
            return jsonify({
                "messages_used": 0,
                "messages_remaining": RATE_LIMIT_MAX,
                "limit": RATE_LIMIT_MAX,
                "window_hours": 4,
                "reset_in": RATE_LIMIT_WINDOW
            })
        count = int(count)
        ttl = redis_client.ttl(key)
        reset_in = ttl if ttl > 0 else RATE_LIMIT_WINDOW
        hours = reset_in // 3600
        mins = (reset_in % 3600) // 60
        return jsonify({
            "messages_used": count,
            "messages_remaining": max(0, RATE_LIMIT_MAX - count),
            "limit": RATE_LIMIT_MAX,
            "window_hours": 4,
            "reset_in": reset_in,
            "reset_in_str": f"{hours}h {mins}m"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/kafka/status", methods=["GET"])
def kafka_status():
    return jsonify({
        "kafka_enabled": KAFKA_ENABLED,
        "broker": KAFKA_BROKER,
        "topic": KAFKA_TOPIC,
        "mode": "Kafka Queue" if KAFKA_ENABLED else "Direct Processing"
    })


    
@app.route("/upload", methods=["POST"])
@admin_required
def upload_file():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file provided."})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."})
    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Only PDF, TXT, and image files allowed!"})
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        existing = ProcessedFile.query.filter_by(user_id=user_id, filename=filename).first()
        if existing:
            return jsonify({
                "success": False,
                "message": f"⚠️ You already uploaded '{filename}'! Please rename and try again.",
                "file_exists": True
            })
        file.save(filepath)
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in OCR_SUPPORTED:
            ocr_path, error = save_ocr_text_to_file(filepath, app.config["UPLOAD_FOLDER"])
            if error:
                return jsonify({"success": False, "message": f"OCR failed: {error}"})
            filename = os.path.basename(ocr_path)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        ingest_documents()
        reload_vectorstore()
        save_processed_file_info(user_id, filename, filepath, chunk_count=1)
        return jsonify({
            "success": True,
            "message": f"✅ '{filename}' uploaded and processed successfully!",
            "is_ocr": file_ext in OCR_SUPPORTED
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/documents", methods=["GET"])
@admin_required
def list_documents():
    user_id = session.get('user_id')
    try:
        user_files = ProcessedFile.query.filter_by(user_id=user_id).all()
        files = []
        for file_record in user_files:
            files.append({
                "file_id": file_record.file_id if hasattr(file_record, 'file_id') else file_record.id,
                "name": file_record.filename,
                "size": f"{file_record.file_size/1024:.1f} KB" if file_record.file_size and file_record.file_size < 1024*1024
                        else f"{file_record.file_size/1024/1024:.1f} MB" if file_record.file_size else "N/A",
                "type": file_record.filename.rsplit('.', 1)[1].upper(),
                "uploaded": str(file_record.processed_at),
                "chunks": file_record.chunk_count,
                "version": file_record.version
            })
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/admin/file-ids", methods=["GET"])
@admin_required
def get_file_ids():
    try:
        files = ProcessedFile.query.all()
        mapping = [{
            "file_id": f.file_id,
            "filename": f.filename,
            "file_type": f.filename.rsplit('.', 1)[1].upper() if '.' in f.filename else 'Unknown',
            "uploaded": str(f.processed_at)
        } for f in files]
        return jsonify({"success": True, "files": mapping})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
@app.route("/documents/<filename>", methods=["DELETE"])
@admin_required
def delete_document(filename):
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file_record = ProcessedFile.query.filter_by(filename=filename).first()
        if not file_record:
            return jsonify({"success": False, "message": "File not found in database."})

        # ✅ FIX 3: Fetch chunk IDs from Redis FIRST before any flush
        chunk_ids = get_chunk_ids_for_file(filename)
        print(f"📋 Pre-fetched {len(chunk_ids)} chunk IDs for: {filename}")

        # ✅ STEP 1: Delete physical files
        for fp in [
            filepath,
            filepath.rsplit('.', 1)[0] + '_ocr.txt',
            filepath.rsplit('.', 1)[0] + '_vision.txt'
        ]:
            if os.path.exists(fp):
                os.remove(fp)
                print(f"🗑️ Deleted file: {fp}")

        # ✅ STEP 2: Delete from ProcessedFile table + add to blacklist
        db.session.delete(file_record)
        add_to_blacklist(filename)

        # ✅ STEP 3: Delete from Neo4j graph
        try:
            delete_graph_for_file(filename)
        except Exception as e:
            print(f"⚠️ Graph cleanup warning: {e}")

        # ✅ STEP 4: Delete the source index key from Redis (before flush)
        delete_source_index(filename)

        # ✅ STEP 5: NOW flush Redis (after we've already fetched chunk_ids above)
        try:
            redis_client.flushdb()
            # Re-add blacklist since flushdb clears everything
            add_to_blacklist(filename)
            print("🗑️ Cleared ALL Redis cache and re-added blacklist entry")
        except Exception as e:
            print(f"⚠️ Redis flush warning: {e}")

        # ✅ STEP 6: Clear semantic cache (PostgreSQL) — single call, no duplicate
        try:
            SemanticCacheRecord.query.delete()
            print("🗑️ Cleared ALL semantic cache (PostgreSQL)")
        except Exception as e:
            print(f"⚠️ Semantic cache cleanup warning: {e}")

        # ✅ STEP 7: Clear FAISS semantic cache store
        try:
            if os.path.exists("semantic_cache_store"):
                shutil.rmtree("semantic_cache_store")
                print("🗑️ Cleared FAISS semantic cache store")
        except Exception as e:
            print(f"⚠️ FAISS semantic cache cleanup warning: {e}")

        db.session.commit()

        # ✅ STEP 8: FAISS chunk deletion using pre-fetched chunk_ids
        try:
            if os.path.exists("vectorstore"):
                vs = FAISS.load_local("vectorstore", embeddings,
                                      allow_dangerous_deserialization=True)
                base_name = filename.rsplit('.', 1)[0].lower()
                ids_to_delete = []

                if chunk_ids:
                    # Primary strategy: match by chunk_index + source name
                    chunk_ids_int = [int(cid) for cid in chunk_ids if str(cid).isdigit()]
                    for doc_id, doc_idx in vs.index_to_docstore_id.items():
                        doc = vs.docstore.search(doc_idx)
                        if doc:
                            chunk_index = doc.metadata.get("chunk_index")
                            source = doc.metadata.get("source", "").lower()
                            if chunk_index in chunk_ids_int and base_name in source:
                                ids_to_delete.append(doc_idx)

                    if not ids_to_delete:
                        # Fallback: match by source name only
                        print(f"⚠️ Chunk ID match found 0 results — falling back to source name scan")
                        for doc_id, doc_idx in vs.index_to_docstore_id.items():
                            doc = vs.docstore.search(doc_idx)
                            if doc and base_name in doc.metadata.get("source", "").lower():
                                ids_to_delete.append(doc_idx)
                else:
                    # No chunk IDs available — scan by source name only
                    print(f"⚠️ No chunk IDs found in Redis — scanning FAISS by source name")
                    for doc_id, doc_idx in vs.index_to_docstore_id.items():
                        doc = vs.docstore.search(doc_idx)
                        if doc and base_name in doc.metadata.get("source", "").lower():
                            ids_to_delete.append(doc_idx)

                if ids_to_delete:
                    vs.delete(ids_to_delete)
                    vs.save_local("vectorstore")
                    print(f"✅ Deleted {len(ids_to_delete)} FAISS chunks for: {filename}")
                else:
                    print(f"⚠️ No matching FAISS chunks found for: {filename}")

        except Exception as e:
            print(f"⚠️ FAISS deletion error: {e}")

        # ✅ STEP 9: Clear record manager cache
        try:
            if os.path.exists("record_manager_cache.db"):
                import sqlite3
                conn = sqlite3.connect("record_manager_cache.db")
                cursor = conn.cursor()
                base_name = filename.rsplit('.', 1)[0].lower()
                cursor.execute("""
                    DELETE FROM upsertion_record
                    WHERE source_id LIKE ? OR source_id LIKE ?
                    OR source_id LIKE ? OR source_id LIKE ?
                """, (f"%{filename}%", f"%{base_name}_ocr%",
                      f"%{base_name}_vision%", f"%{base_name}.txt%"))
                conn.commit()
                conn.close()
                print(f"✅ Cleared record manager for: {filename}")
        except Exception as e:
            print(f"⚠️ Record manager cleanup: {e}")

        # ✅ STEP 10: Reload vectorstore
        reload_vectorstore()

        return jsonify({
            "success": True,
            "message": f"🗑️ '{filename}' and ALL related cache/data deleted successfully!"
        })

    except Exception as e:
        db.session.rollback()
        print(f"❌ Delete error: {e}")
        return jsonify({"success": False, "message": str(e)})

# ============================================================
# --- AUTH ROUTES ---
# ============================================================

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    if not username or not email or not password:
        return jsonify({"success": False, "message": "All fields required!"})
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be 6+ characters!"})
    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "message": "Username already exists!"})
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered!"})
    try:
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        session['is_admin'] = new_user.is_admin
        return jsonify({"success": True, "message": f"Welcome {username}! 🎉", "user": new_user.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required!"})
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"success": False, "message": "Invalid username or password!"})
    session['user_id'] = user.id
    session['username'] = user.username
    session['is_admin'] = user.is_admin
    return jsonify({"success": True, "message": f"Welcome back, {username}! 👋", "user": user.to_dict()})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully!"})

@app.route("/current-user", methods=["GET"])
def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Not logged in"})
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"success": False, "message": "User not found"})
    return jsonify({"success": True, "user": user.to_dict()})

@app.route('/view_document/<filename>', methods=['GET'])
def view_document(filename):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except FileNotFoundError:
        abort(404)

# ============================================================
# --- DASHBOARD ROUTES ---
# ============================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login first!"}), 401
    try:
        results = db.session.execute(db.text("""
            SELECT
                u.username,
                user_msg.content as question,
                asst_msg.content as answer,
                asst_msg.cache_source,
                asst_msg.response_time_ms,
                re.score,
                re.feedback,
                user_msg.created_at
            FROM chat_history user_msg
            JOIN users u ON user_msg.user_id = u.id
            LEFT JOIN chat_history asst_msg
                ON asst_msg.session_id = user_msg.session_id
                AND asst_msg.role = 'assistant'
                AND asst_msg.created_at > user_msg.created_at
            LEFT JOIN response_evaluations re
                ON re.user_id = user_msg.user_id
                AND re.question = user_msg.content
            WHERE user_msg.role = 'user'
            ORDER BY user_msg.created_at DESC
            LIMIT 100
        """)).fetchall()

        stats = db.session.execute(db.text("""
            SELECT
                (SELECT COUNT(*) FROM chat_history WHERE role = 'user') as total_queries,
                (SELECT ROUND(AVG(score), 2) FROM response_evaluations) as avg_score,
                (SELECT COUNT(*) FROM chat_history
                 WHERE role = 'assistant'
                 AND cache_source IS NOT NULL
                 AND cache_source NOT IN ('NONE', '')) as cache_hits,
                ROUND(
                    (SELECT COUNT(*) FROM chat_history
                     WHERE role = 'assistant'
                     AND cache_source IS NOT NULL
                     AND cache_source NOT IN ('NONE', '')) * 100.0 /
                    NULLIF((SELECT COUNT(*) FROM chat_history WHERE role = 'assistant'), 0),
                    2
                ) as cache_hit_rate
        """)).fetchone()

        cache_breakdown = db.session.execute(db.text("""
            SELECT
                CASE
                    WHEN cache_source LIKE '%REDIS%' THEN 'REDIS'
                    WHEN cache_source LIKE '%SEMANTIC%' THEN 'SEMANTIC CACHE'
                    ELSE 'FULL PIPELINE'
                END as cache_source,
                COUNT(*) as count,
                ROUND(AVG(response_time_ms), 0) as avg_ms
            FROM chat_history
            WHERE role = 'assistant'
            AND cache_source IS NOT NULL
            GROUP BY
                CASE
                    WHEN cache_source LIKE '%REDIS%' THEN 'REDIS'
                    WHEN cache_source LIKE '%SEMANTIC%' THEN 'SEMANTIC CACHE'
                    ELSE 'FULL PIPELINE'
                END
            ORDER BY count DESC
        """)).fetchall()

        return jsonify({
            "success": True,
            "data": [{
                "username": r.username,
                "question": r.question,
                "answer": r.answer,
                "cache_source": r.cache_source,
                "response_time_ms": r.response_time_ms,
                "score": r.score,
                "feedback": r.feedback,
                "created_at": str(r.created_at)
            } for r in results],
            "stats": {
                "total_queries": stats.total_queries or 0,
                "avg_score": float(stats.avg_score) if stats.avg_score else 0,
                "cache_hits": stats.cache_hits or 0,
                "cache_hit_rate": float(stats.cache_hit_rate) if stats.cache_hit_rate else 0
            },
            "cache_breakdown": [{
                "cache_source": c.cache_source,
                "count": c.count,
                "avg_ms": float(c.avg_ms) if c.avg_ms else 0
            } for c in cache_breakdown]
        })
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

@app.route("/admin-dashboard")
def admin_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/')
    user = User.query.get(user_id)
    if not user or not user.is_admin:
        return redirect('/')
    return render_template("dashboard.html")

@app.route("/dashboard-view")
def dashboard_view():
    if not session.get('user_id'):
        return redirect('/')
    return render_template("dashboard.html")

# ============================================================
# --- ADMIN DOCUMENT ROUTES ---
# ============================================================

@app.route("/admin/documents", methods=["GET"])
@admin_required
def admin_documents():
    try:
        files = db.session.execute(db.text("""
            SELECT
                pf.file_id,
                pf.filename,
                u.username,
                pf.file_size,
                pf.chunk_count,
                pf.version,
                pf.processed_at,
                pf.file_hash
            FROM processed_files pf
            JOIN users u ON pf.user_id = u.id
            ORDER BY pf.processed_at DESC
        """)).fetchall()
        stats = db.session.execute(db.text("""
            SELECT
                COUNT(*) as total_files,
                SUM(file_size) as total_size,
                SUM(chunk_count) as total_chunks
            FROM processed_files
        """)).fetchone()
        return jsonify({
            "success": True,
            "files": [{
                "file_id": f.file_id,
                "filename": f.filename,
                "username": f.username,
                "file_size": f.file_size,
                "size_str": f"{f.file_size/1024:.1f} KB" if f.file_size and f.file_size < 1024*1024
                            else f"{f.file_size/1024/1024:.1f} MB" if f.file_size else "N/A",
                "chunk_count": f.chunk_count,
                "version": f.version,
                "processed_at": str(f.processed_at),
                "file_type": f.filename.rsplit('.', 1)[1].upper() if '.' in f.filename else 'Unknown'
            } for f in files],
            "stats": {
                "total_files": stats.total_files or 0,
                "total_size": f"{stats.total_size/1024/1024:.1f} MB" if stats.total_size else "0 MB",
                "total_chunks": stats.total_chunks or 0
            }
        })
    except Exception as e:
        print(f"❌ Admin documents error: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route("/admin/documents-view")
def admin_documents_view():
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/')
    user = User.query.get(user_id)
    if not user or not user.is_admin:
        return redirect('/')
    return render_template("admin_documents.html")

# ============================================================
# --- ADMIN FEEDBACK ROUTES ---
# ============================================================

@app.route("/admin/feedback", methods=["POST"])
@admin_required
def submit_admin_feedback():
    admin_id = session.get('user_id')
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    original_answer = data.get("original_answer", "").strip()
    feedback_type = data.get("feedback_type", "").strip()
    user_id = data.get("user_id")
    correct_answer = data.get("correct_answer", "").strip()

    if not question or not original_answer:
        return jsonify({"success": False, "message": "Missing data!"})
    if feedback_type not in ['thumbs_up', 'thumbs_down']:
        return jsonify({"success": False, "message": "Invalid feedback type!"})
    if feedback_type == 'thumbs_down' and not correct_answer:
        return jsonify({"success": False, "message": "Please provide correct answer!"})

    try:
        feedback = AdminFeedback(
            user_id=user_id or admin_id,
            admin_id=admin_id,
            question=question,
            original_answer=original_answer,
            feedback_type=feedback_type,
            correct_answer=correct_answer if feedback_type == 'thumbs_down' else None
        )
        db.session.add(feedback)
        db.session.commit()

        if feedback_type == 'thumbs_down' and correct_answer:
            ingest_success = ingest_feedback_to_vectorstore(question, correct_answer, feedback.id)
            if ingest_success:
                feedback.is_ingested = True
                db.session.commit()
                cache_key = f"rag:{question.lower().strip()}"
                redis_client.setex(cache_key, REDIS_TTL, json.dumps(correct_answer))
                print(f"✅ Correct answer ingested to vectorstore!")
                return jsonify({
                    "success": True,
                    "message": "👎 Feedback saved & correct answer added to knowledge base!"
                })

        print(f"{'👍' if feedback_type == 'thumbs_up' else '👎'} Admin feedback saved!")
        return jsonify({
            "success": True,
            "message": f"{'👍 Marked as correct!' if feedback_type == 'thumbs_up' else '👎 Feedback saved!'}"
        })
    except Exception as e:
        db.session.rollback()
        print(f"❌ Feedback error: {e}")
        return jsonify({"success": False, "message": str(e)})
    


@app.route("/admin/feedback/list", methods=["GET"])
@admin_required
def get_admin_feedback():
    try:
        feedbacks = db.session.execute(db.text("""
            SELECT
                af.id,
                u.username as user_name,
                a.username as admin_name,
                af.question,
                af.original_answer,
                af.feedback_type,
                af.correct_answer,
                af.is_ingested,
                af.created_at
            FROM admin_feedback af
            JOIN users u ON af.user_id = u.id
            JOIN users a ON af.admin_id = a.id
            ORDER BY af.created_at DESC
        """)).fetchall()
        return jsonify({
            "success": True,
            "feedbacks": [{
                "id": f.id,
                "username": f.user_name,
                "admin_name": f.admin_name,
                "question": f.question,
                "original_answer": f.original_answer,
                "feedback_type": f.feedback_type,
                "correct_answer": f.correct_answer,
                "is_ingested": f.is_ingested,
                "created_at": str(f.created_at)
            } for f in feedbacks]
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ============================================================
# --- SCHEDULER ---
# ============================================================

def scheduled_cleanup():
    with app.app_context():
        cleanup_expired_cache()

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_cleanup, 'interval', hours=4)
scheduler.start()
print("✅ Cache cleanup scheduler started (runs every 4 hours)")

@app.route("/admin/feedback/<int:feedback_id>/remove", methods=["DELETE"])
@admin_required
def remove_feedback(feedback_id):
    """Remove a feedback entry and its learned answer from vectorstore"""
    try:
        feedback = AdminFeedback.query.get(feedback_id)
        if not feedback:
            return jsonify({"success": False, "message": "Feedback not found!"})

        source_id = f"admin_feedback_{feedback_id}"

        global vectorstore, ALL_DOCS, bm25_retriever, dense_retriever

        if vectorstore:
            ALL_DOCS = [doc for doc in ALL_DOCS
                       if doc.metadata.get("source") != source_id]

            if ALL_DOCS:
                vectorstore = FAISS.from_documents(ALL_DOCS, embeddings)
                vectorstore.save_local("vectorstore")
                dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
            else:
                dummy_doc = Document(
                    page_content="No documents uploaded yet.",
                    metadata={"source": "system", "filetype": "txt", "chunk_index": 0}
                )
                vectorstore = FAISS.from_documents([dummy_doc], embeddings)
                vectorstore.save_local("vectorstore")
                ALL_DOCS = [dummy_doc]

            print(f"✅ Removed feedback_{feedback_id} from vectorstore!")

        cache_key = f"rag:{feedback.question.lower().strip()}"
        redis_client.delete(cache_key)
        print(f"✅ Removed from Redis cache!")

        SemanticCacheRecord.query.filter_by(
            query_text=feedback.question
        ).delete()
        db.session.commit()
        print(f"✅ Removed from semantic cache!")

        db.session.delete(feedback)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": f"✅ Wrong answer removed from all caches and vectorstore!"
        })

    except Exception as e:
        db.session.rollback()
        print(f"❌ Remove feedback error: {e}")
        return jsonify({"success": False, "message": str(e)})

# ============================================================
# --- MAIN ---
# ============================================================

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        sync_existing_documents()
    reload_vectorstore()
    app.run(debug=True)