import os
import uuid
import time
import json
import redis
import shutil
import hashlib
import re
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from functools import wraps
from flask import Flask, request, jsonify, render_template
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
from ingest import ingest_documents
from dotenv import load_dotenv
from sqlalchemy import func
 
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
app = Flask(__name__)
app.secret_key = "123456"
# --- FILE UPLOAD CONFIG ---
UPLOAD_FOLDER = "documents"
ALLOWED_EXTENSIONS = {"pdf", "txt", "jpg", "jpeg", "png", "bmp", "gif"}
OCR_SUPPORTED = {"jpg", "jpeg", "png", "bmp", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
 
# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# --- POSTGRESQL CONFIG ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/postgres'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
 
# --- REDIS CONFIG ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDIS_TTL = 3600
 
# --- CHAT HISTORY MODEL ---
class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # ✅ NEW
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
# --- USER MODEL ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    
    # Relationship to chat history
    chats = db.relationship('ChatHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": str(self.created_at)
        }
    
# --- PROCESSED FILES TRACKING MODEL ---
class ProcessedFile(db.Model):
    __tablename__ = 'processed_files'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # ✅ NEW
    filename = db.Column(db.String(255), nullable=False)  # ✅ REMOVED unique=True
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
    
# --- DB HELPERS ---
def save_chat_message(user_id, session_id, role, content, cache_source='NONE', response_time_ms=0):
    try:
        msg = ChatHistory(
            user_id=user_id,  # ✅ NEW
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
 
# --- EMBEDDINGS & VECTOR STORE SAFE INITIALIZATION ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
def initialize_vectorstore():
    """Safely loads the vectorstore, or creates a placeholder if empty."""
    if os.path.exists("vectorstore") and os.path.exists(os.path.join("vectorstore", "index.faiss")):
        try:
            vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            # Safely recover true internal document shards
            docs = [vs.docstore.search(idx) for idx in vs.index_to_docstore_id.values()]
            # Filter out pure setup labels
            valid_docs = [d for d in docs if d.metadata.get("source") != "system"]
            if valid_docs:
                print(f"📚 System successfully mapped {len(valid_docs)} document pieces into the runtime index context.")
                return vs, valid_docs
        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}. Reinitializing.")
            
    print("⚠️ No valid document data discovered yet. Standing by for upload files...")
    dummy_doc = Document(page_content="No documents have been uploaded yet.", metadata={"source": "system", "filetype": "txt", "chunk_index": 0})
    vs = FAISS.from_documents([dummy_doc], embeddings)
    return vs, [dummy_doc]
 
vectorstore, ALL_DOCS = initialize_vectorstore()
 
# --- HYBRID RETRIEVER SETUP ---
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
 
RRF_K = 60
 
def reciprocal_rank_fusion(bm25_docs, dense_docs):
    """Merge BM25 and FAISS results using Reciprocal Rank Fusion."""
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
    """Run BM25 + FAISS in parallel, merge with RRF."""
    bm25_docs = bm25_retriever.invoke(question)
    dense_docs = dense_retriever.invoke(question)
    return reciprocal_rank_fusion(bm25_docs, dense_docs)
 
# --- FLASHRANK RERANKER ---
reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="reranker_cache")
 
def rerank_documents(question, docs, top_n=3):
    """Rerank retrieved docs and return top N most relevant."""
    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
        return docs[:top_n]
    
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = reranker.rerank(rerank_request)
 
    top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    top_indices = [r["id"] for r in top_results]
    return [docs[i] for i in top_indices]
 
# --- GROQ LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))
 
# --- REDIS CACHE HELPERS ---
def check_redis_cache(question):
    try:
        cache_key = f"rag:{question.lower().strip()}"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"⚠️ Redis read error: {e}")
    return None
 
def save_to_redis_cache(question, response_text):
    try:
        cache_key = f"rag:{question.lower().strip()}"
        redis_client.setex(cache_key, REDIS_TTL, json.dumps(response_text))
    except Exception as e:
        print(f"⚠️ Redis write error: {e}")
 
# --- SEMANTIC CACHE (FAISS) ---
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
    new_doc = Document(
        page_content=question,
        metadata={"response": response_text}
    )
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
 
# --- FILE UPLOAD HELPERS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def reload_vectorstore():
    """Reloads global vector references using the safe initialization schema."""
    global vectorstore, dense_retriever, bm25_retriever, ALL_DOCS
    vectorstore, ALL_DOCS = initialize_vectorstore()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(ALL_DOCS, k=6)
    print(f"🔄 Vectorstore configuration reloaded globally.")

def is_casual_query(question):
    """Determine if a query is chit-chat/greeting or requires document search."""
    # Fast local regex match for common greetings
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|wassup|yo|who are you|what is your name)\b"
    if re.search(greetings, question.lower().strip()):
        return True
    return False
# --- MAIN ANSWER FUNCTION ---
def get_answer(question, session_id):
    """
    Get answer with 3 return values:
    - answer (string)
    - cache_source (string): REDIS, SEMANTIC, or NONE
    - retrieval_info (string): Alpha parameter info
    """
    # 🌟 NEW: Check for casual greetings/conversational intent first
    if is_casual_query(question):
        print("💬 Casual query detected — routing directly to LLM conversational mode.")
        prompt = f"""You are a helpful, professional, and friendly AI assistant. 
Respond to the user's greeting or casual statement naturally and warmly. Keep it concise.

### USER MESSAGE:
{question}

Answer:"""
        response = llm.invoke(prompt)
        return response.content, "NONE", "⚡ Direct LLM Conversation (No Document Search)"

    # 1. Check Redis cache (exact match) - Rest of your code continues below normally...
    
    cached = check_redis_cache(question)
    if cached:
        return cached, "REDIS", ""
 
    cached = check_semantic_cache(question)
    if cached:
        save_to_redis_cache(question, cached)
        return cached, "SEMANTIC", ""
 
    docs = hybrid_retrieve(question)
    docs = rerank_documents(question, docs, top_n=3)
 
    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
        return "I don't have any uploaded documents to extract information from right now.", "NONE", ""
 
    context = ""
    sources_set = set()
    
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filetype = doc.metadata.get("filetype", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        sources_set.add(source)
        context += f"[Source: {source} | Type: {filetype} | Chunk: {chunk_index}]\n"
        context += doc.page_content + "\n\n"
 
    alpha = 0.7
    retrieval_info = f"📊 Retrieved using: {int(alpha*100)}% Semantic (FAISS) + {int((1-alpha)*100)}% Keyword (BM25)\n📎 Sources: {', '.join(sources_set)}"
    past_history = get_past_chat_history(session_id, limit=6)
 
    prompt = f"""You are an expert document assistant. You are provided with OCR text from a document.
      You must answer ONLY using the provided OCR text. If the document mentions a year, use that year. 
      Do not use the current date or your internal knowledge of the year 2026 to answer questions. 
      If the information is not in the text, state that you do not have the information
 
### PAST CONVERSATION:
{past_history}
 
### CONTEXT:
{context}
 
### QUESTION:
{question}
 
Answer:"""
 
    response = llm.invoke(prompt)
    answer = response.content
 
    save_to_redis_cache(question, answer)
    save_to_semantic_cache(question, answer)
 
    return answer, "NONE", retrieval_info

# --- FILE HASH & PROCESSING HELPERS ---
def calculate_file_hash(filepath):
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_file_version(filename):
    match = re.search(r'(_v\d+|_updated|_v\d+_updated)', filename.lower())
    return match.group(1) if match else "v1"

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
                user_id=user_id,  # ✅ NEW
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

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/chat", methods=["POST"])
def chat():
    # ✅ CHECK IF USER IS LOGGED IN
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
 
    try:
        start_time = time.time()
        save_chat_message(user_id, session_id, "user", user_message)  # ✅ Pass user_id
        reply, cache_source, retrieval_info = get_answer(user_message, session_id)
        duration = int((time.time() - start_time) * 1000)
        save_chat_message(user_id, session_id, "assistant", reply, cache_source=cache_source, response_time_ms=duration)  # ✅ Pass user_id
        
        return jsonify({
            "reply": reply,
            "retrieval_info": retrieval_info,
            "cache_source": cache_source,
            "duration_ms": duration,
            "user": user.username
        })
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})
 
@app.route("/upload", methods=["POST"])
def upload_file():
    # ✅ CHECK IF USER IS LOGGED IN
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login first!"})
    
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"success": False, "message": "User not found!"})
    
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
        
        # ✅ CHECK IF THIS USER ALREADY HAS THIS FILE
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
        
        # ✅ SAVE WITH USER_ID
        save_processed_file_info(user_id, filename, filepath, chunk_count=1)
        
        return jsonify({
            "success": True, 
            "message": f"✅ '{filename}' uploaded and processed successfully!",
            "is_ocr": file_ext in OCR_SUPPORTED
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
@app.route("/documents", methods=["GET"])
def list_documents():
    # ✅ CHECK IF USER IS LOGGED IN
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login first!"})
    
    try:
        # ✅ GET ONLY THIS USER'S FILES FROM DATABASE
        user_files = ProcessedFile.query.filter_by(user_id=user_id).all()
        
        files = []
        for file_record in user_files:
            files.append({
                "name": file_record.filename,
                "size": f"{file_record.file_size/1024:.1f} KB" if file_record.file_size < 1024*1024 else f"{file_record.file_size/1024/1024:.1f} MB",
                "type": file_record.filename.rsplit('.', 1)[1].upper(),
                "uploaded": str(file_record.processed_at),
                "chunks": file_record.chunk_count,
                "version": file_record.version
            })
        
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
 
@app.route("/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    # ✅ CHECK IF USER IS LOGGED IN
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login first!"})
    
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"success": False, "message": "User not found!"})
    
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # ✅ CHECK IF THIS FILE BELONGS TO THIS USER
        file_record = ProcessedFile.query.filter_by(user_id=user_id, filename=filename).first()
        if not file_record:
            return jsonify({"success": False, "message": "File not found or you don't have permission to delete it."})
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": "File not found on disk."})

        os.remove(filepath)
        
        # ✅ DELETE FROM DATABASE
        db.session.delete(file_record)
        db.session.commit()
        
        # Clear Redis cache for this file
        redis_client.delete(f"fastpass_hash:{filename}")
        
        # Re-ingest remaining documents
        remaining = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
        if not remaining:
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            if os.path.exists("record_manager_cache.db"):
                os.remove("record_manager_cache.db")
        else:
            ingest_documents()

        reload_vectorstore()
        return jsonify({"success": True, "message": f"🗑️ '{filename}' deleted successfully!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)})
    
# --- AUTHENTICATION ROUTES ---
# --- AUTHENTICATION ROUTES ---

@app.route("/signup", methods=["POST"])
def signup():
    """Register a new user."""
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
        
        return jsonify({
            "success": True,
            "message": f"Welcome {username}! 🎉",
            "user": new_user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@app.route("/login", methods=["POST"])
def login():
    """Login an existing user."""
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
    
    return jsonify({
        "success": True,
        "message": f"Welcome back, {username}! 👋",
        "user": user.to_dict()
    })


@app.route("/logout", methods=["POST"])
def logout():
    """Logout user."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully!"})


@app.route("/current-user", methods=["GET"])
def current_user():
    """Get currently logged in user."""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({"success": False, "message": "Not logged in"})
    
    user = User.query.get(user_id)
    
    if not user:
        session.clear()
        return jsonify({"success": False, "message": "User not found"})
    
    return jsonify({"success": True, "user": user.to_dict()})
from flask import send_from_directory, abort

@app.route('/view_document/<filename>', methods=['GET'])
def view_document(filename):
    # Ensure the user is logged in before viewing files
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    try:
        # 'documents' is your UPLOAD_FOLDER
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except FileNotFoundError:
        abort(404)
 
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # Force runtime validation mapping check right at execution launch
    reload_vectorstore()
    app.run(debug=True)     