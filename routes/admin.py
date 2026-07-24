import os
import shutil
from datetime import datetime
from flask import Blueprint, request, jsonify, session, redirect, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from functools import wraps
from extensions import db, redis_client
from models.models import User, ProcessedFile, SemanticCacheRecord, AdminFeedback, ChatHistory, ResponseEvaluation
from services.cache import (
    add_to_blacklist, get_chunk_ids_for_file,
    delete_source_index
)
from services.rag import ingest_feedback_to_vectorstore
from services.vectorstore import reload_vectorstore, archive_file_chunks
from ingest import ingest_documents
from graph_handler import delete_graph_for_file
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, OCR_SUPPORTED, REDIS_TTL
from extensions import embeddings

admin_bp = Blueprint('admin', __name__)

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
            return jsonify({"success": False, "message": "User not found!"}), 401
        if not user.is_admin:
            return jsonify({
                "success": False,
                "message": "⛔ Admin access required!",
                "error": "not_admin"
            }), 403
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================
# --- DOCUMENT ROUTES ---
# ============================================================

@admin_bp.route("/upload", methods=["POST"])
@admin_required
def upload_file():
    from ocr_handler import save_ocr_text_to_file
    user_id = session.get('user_id')
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file provided."})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."})
    print(f"🔍 DEBUG: filename={file.filename}, ALLOWED_EXTENSIONS={ALLOWED_EXTENSIONS}")
    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Only PDF files are allowed!"})
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
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
            ocr_path, error = save_ocr_text_to_file(filepath, UPLOAD_FOLDER)
            if error:
                return jsonify({"success": False, "message": f"OCR failed: {error}"})
            filename = os.path.basename(ocr_path)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

        # ✅ Publish to Kafka instead of processing inline. A separate
        # ingest_worker.py process pool handles it, so this request
        # returns immediately — 10 simultaneous uploads don't queue
        # behind each other, each gets a real OS process.
        from kafka_handler import send_ingestion_request
        request_id = send_ingestion_request(user_id, filename, filepath)

        if request_id is None:
            # Kafka unavailable — fall back to the old synchronous path
            chunk_counts = ingest_documents(filename)
            if not chunk_counts:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "success": False,
                    "message": f"❌ '{filename}' could not be processed. Only valid PDF files are supported."
                })
            reload_vectorstore()
            real_count = chunk_counts.get(filename, 0)
            save_processed_file_info(user_id, filename, filepath, chunk_count=real_count)
            return jsonify({
                "success": True,
                "message": f"✅ '{filename}' uploaded and processed successfully!",
                "done": True
            })

        return jsonify({
            "success": True,
            "message": f"📤 '{filename}' queued for processing...",
            "request_id": request_id,
            "done": False
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@admin_bp.route("/documents", methods=["GET"])
@admin_required
def list_documents():
    user_id = session.get('user_id')
    try:
        user_files = ProcessedFile.query.filter_by(user_id=user_id).all()
        files = []
        for file_record in user_files:
            files.append({
                "file_id": file_record.file_id,
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

@admin_bp.route("/upload/result/<request_id>", methods=["GET"])
@admin_required
def upload_result(request_id):
    import json
    result_json = redis_client.get(f"ingest_result:{request_id}")
    if result_json is None:
        return jsonify({"done": False})
    result = json.loads(result_json)
    return jsonify({"done": True, **result})

@admin_bp.route("/documents/<filename>", methods=["DELETE"])
@admin_required
def delete_document(filename):
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file_record = ProcessedFile.query.filter_by(filename=filename).first()
        if not file_record:
            return jsonify({"success": False, "message": "File not found in database."})

        chunk_ids = get_chunk_ids_for_file(filename)
        print(f"📋 Pre-fetched {len(chunk_ids)} chunk IDs for: {filename}")

        for fp in [filepath,
                   filepath.rsplit('.', 1)[0] + '_ocr.txt',
                   filepath.rsplit('.', 1)[0] + '_vision.txt']:
            if os.path.exists(fp):
                os.remove(fp)
                print(f"🗑️ Deleted file: {fp}")

        stored_file_id = file_record.file_id

        db.session.delete(file_record)

        try:
            delete_graph_for_file(filename)
        except Exception as e:
            print(f"⚠️ Graph cleanup warning: {e}")

        delete_source_index(filename)

        # ✅ No longer flushdb() — that wiped chunking decisions, semantic cache,
        # and rate limits for EVERY document, not just the one being deleted.
        # Chunking decisions are content-hash-keyed, so they're unaffected by
        # deleting a specific filename anyway — only blacklist this file.
        try:
            add_to_blacklist(filename)
            print("🗑️ Added file to blacklist")
        except Exception as e:
            print(f"⚠️ Blacklist warning: {e}")

        try:
            SemanticCacheRecord.query.delete()
            print("🗑️ Cleared ALL semantic cache (PostgreSQL)")
        except Exception as e:
            print(f"⚠️ Semantic cache cleanup warning: {e}")

        try:
            if os.path.exists("semantic_cache_store"):
                shutil.rmtree("semantic_cache_store")
                print("🗑️ Cleared FAISS semantic cache store")
        except Exception as e:
            print(f"⚠️ FAISS semantic cache cleanup warning: {e}")

        db.session.commit()

        # ✅ Archive chunks in ChromaDB instead of manually walking a FAISS index
        try:
            if stored_file_id:
                archive_file_chunks(stored_file_id)
        except Exception as e:
            print(f"⚠️ ChromaDB archive error: {e}")

        try:
            if os.path.exists("record_manager_cache.db"):
                import sqlite3
                conn = sqlite3.connect("record_manager_cache.db")
                cursor = conn.cursor()
                base_name = filename.rsplit('.', 1)[0].lower()
                cursor.execute("""
                    DELETE FROM upsertion_record
                    WHERE group_id LIKE ? OR group_id LIKE ?
                    OR group_id LIKE ? OR group_id LIKE ?
                """, (f"%{filename}%", f"%{base_name}_ocr%",
                      f"%{base_name}_vision%", f"%{base_name}.txt%"))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"⚠️ Record manager cleanup: {e}")

        reload_vectorstore()
        redis_client.publish("vectorstore_updates", "reload")  # ✅ notify worker to reload

        return jsonify({
            "success": True,
            "message": f"🗑️ '{filename}' and ALL related cache/data deleted successfully!"
        })

    except Exception as e:
        db.session.rollback()
        print(f"❌ Delete error: {e}")
        return jsonify({"success": False, "message": str(e)})


@admin_bp.route("/admin/documents", methods=["GET"])
@admin_required
def admin_documents():
    try:
        files = db.session.execute(db.text("""
            SELECT pf.file_id, pf.filename, u.username, pf.file_size,
                   pf.chunk_count, pf.version, pf.processed_at, pf.file_hash
            FROM processed_files pf
            JOIN users u ON pf.user_id = u.id
            ORDER BY pf.processed_at DESC
        """)).fetchall()
        stats = db.session.execute(db.text("""
            SELECT COUNT(*) as total_files, SUM(file_size) as total_size,
                   SUM(chunk_count) as total_chunks FROM processed_files
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
        return jsonify({"success": False, "message": str(e)})


# ============================================================
# --- FEEDBACK ROUTES ---
# ============================================================

@admin_bp.route("/admin/feedback", methods=["POST"])
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
                redis_client.setex(cache_key, REDIS_TTL, __import__('json').dumps(correct_answer))
                return jsonify({
                    "success": True,
                    "message": "👎 Feedback saved & correct answer added to knowledge base!"
                })

        return jsonify({
            "success": True,
            "message": f"{'👍 Marked as correct!' if feedback_type == 'thumbs_up' else '👎 Feedback saved!'}"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)})


@admin_bp.route("/admin/feedback/list", methods=["GET"])
@admin_required
def get_admin_feedback():
    try:
        feedbacks = db.session.execute(db.text("""
            SELECT af.id, u.username as user_name, a.username as admin_name,
                   af.question, af.original_answer, af.feedback_type,
                   af.correct_answer, af.is_ingested, af.created_at
            FROM admin_feedback af
            JOIN users u ON af.user_id = u.id
            JOIN users a ON af.admin_id = a.id
            ORDER BY af.created_at DESC
        """)).fetchall()
        return jsonify({
            "success": True,
            "feedbacks": [{
                "id": f.id, "username": f.user_name, "admin_name": f.admin_name,
                "question": f.question, "original_answer": f.original_answer,
                "feedback_type": f.feedback_type, "correct_answer": f.correct_answer,
                "is_ingested": f.is_ingested, "created_at": str(f.created_at)
            } for f in feedbacks]
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@admin_bp.route("/admin/feedback/<int:feedback_id>/remove", methods=["DELETE"])
@admin_required
def remove_feedback(feedback_id):
    try:
        feedback = AdminFeedback.query.get(feedback_id)
        if not feedback:
            return jsonify({"success": False, "message": "Feedback not found!"})

        source_id = f"admin_feedback_{feedback_id}"
        import services.vectorstore as vs_module
        from langchain_community.retrievers import BM25Retriever

        if vs_module.vectorstore:
            vs_module.ALL_DOCS = [doc for doc in vs_module.ALL_DOCS
                                  if doc.metadata.get("source") != source_id]
            try:
                matches = vs_module.vectorstore.get(where={"source": source_id}, include=["metadatas"])
                ids_to_delete = matches.get("ids", [])
                if ids_to_delete:
                    vs_module.vectorstore._collection.delete(ids=ids_to_delete)
                    print(f"✅ Deleted {len(ids_to_delete)} Chroma chunks for feedback source: {source_id}")
            except Exception as e:
                print(f"⚠️ Chroma feedback-doc delete error: {e}")

            if vs_module.ALL_DOCS:
                vs_module.bm25_retriever = BM25Retriever.from_documents(vs_module.ALL_DOCS, k=6)

        cache_key = f"rag:{feedback.question.lower().strip()}"
        redis_client.delete(cache_key)
        SemanticCacheRecord.query.filter_by(query_text=feedback.question).delete()
        db.session.commit()
        db.session.delete(feedback)
        db.session.commit()

        return jsonify({"success": True, "message": "✅ Wrong answer removed from all caches and vectorstore!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)})


# ============================================================
# --- DASHBOARD ROUTES ---
# ============================================================

@admin_bp.route("/dashboard", methods=["GET"])
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Please login first!"}), 401
    try:
        results = db.session.execute(db.text("""
            SELECT u.username, user_msg.content as question,
                   asst_msg.content as answer, asst_msg.cache_source,
                   asst_msg.response_time_ms, re.score, re.feedback, user_msg.created_at,
                   user_msg.ip_address
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
            ORDER BY user_msg.created_at DESC LIMIT 100
        """)).fetchall()

        stats = db.session.execute(db.text("""
            SELECT
                (SELECT COUNT(*) FROM chat_history WHERE role = 'user') as total_queries,
                (SELECT ROUND(AVG(score), 2) FROM response_evaluations) as avg_score,
                (SELECT COUNT(*) FROM chat_history WHERE role = 'assistant'
                 AND cache_source IS NOT NULL AND cache_source NOT IN ('NONE', '')) as cache_hits,
                ROUND((SELECT COUNT(*) FROM chat_history WHERE role = 'assistant'
                 AND cache_source IS NOT NULL AND cache_source NOT IN ('NONE', '')) * 100.0 /
                NULLIF((SELECT COUNT(*) FROM chat_history WHERE role = 'assistant'), 0), 2) as cache_hit_rate
        """)).fetchone()

        cache_breakdown = db.session.execute(db.text("""
            SELECT
                CASE WHEN cache_source LIKE '%REDIS%' THEN 'REDIS'
                     WHEN cache_source LIKE '%SEMANTIC%' THEN 'SEMANTIC CACHE'
                     ELSE 'FULL PIPELINE' END as cache_source,
                COUNT(*) as count, ROUND(AVG(response_time_ms), 0) as avg_ms
            FROM chat_history WHERE role = 'assistant' AND cache_source IS NOT NULL
            GROUP BY CASE WHEN cache_source LIKE '%REDIS%' THEN 'REDIS'
                          WHEN cache_source LIKE '%SEMANTIC%' THEN 'SEMANTIC CACHE'
                          ELSE 'FULL PIPELINE' END
            ORDER BY count DESC
        """)).fetchall()

        return jsonify({
            "success": True,
            "data": [{"username": r.username, "question": r.question, "answer": r.answer,
                      "cache_source": r.cache_source, "response_time_ms": r.response_time_ms,
                      "score": r.score, "feedback": r.feedback, "created_at": str(r.created_at),
                      "ip_address": r.ip_address}
                     for r in results],
            "stats": {
                "total_queries": stats.total_queries or 0,
                "avg_score": float(stats.avg_score) if stats.avg_score else 0,
                "cache_hits": stats.cache_hits or 0,
                "cache_hit_rate": float(stats.cache_hit_rate) if stats.cache_hit_rate else 0
            },
            "cache_breakdown": [{"cache_source": c.cache_source, "count": c.count,
                                  "avg_ms": float(c.avg_ms) if c.avg_ms else 0}
                                 for c in cache_breakdown]
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ============================================================
# --- VIEW DOCUMENT ---
# ============================================================

@admin_bp.route('/view_document/<filename>', methods=['GET'])
def view_document(filename):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        abort(404)


# ============================================================
# --- HELPER ---
# ============================================================

def save_processed_file_info(user_id, filename, filepath, chunk_count):
    try:
        stat = os.stat(filepath)
        file_hash = f"{stat.st_mtime_ns}:{stat.st_size}"
        file_size = os.path.getsize(filepath)
        import re as _re
        match = _re.search(r'(_v\d+|_updated|_v\d+_updated)', filename.lower())
        version = match.group(1) if match else "v1"

        # ✅ Use the SAME file_id that ingest.py already generated and
        # embedded into every chunk's metadata in Chroma — don't let
        # ProcessedFile's default lambda invent a second, different UUID.
        # Without this, Postgres and Chroma disagree on file_id, and
        # delete_document()'s archive_file_chunks() call silently finds
        # nothing to archive.
        chroma_file_id = redis_client.get(f"file_id:{filename}")

        existing = ProcessedFile.query.filter_by(user_id=user_id, filename=filename).first()
        if existing:
            existing.file_hash = file_hash
            existing.file_size = file_size
            existing.chunk_count = chunk_count
            existing.version = version
            existing.processed_at = db.func.now()
            if chroma_file_id:
                existing.file_id = chroma_file_id
        else:
            processed = ProcessedFile(
                user_id=user_id, filename=filename, file_hash=file_hash,
                file_size=file_size, chunk_count=chunk_count, version=version,
                file_id=chroma_file_id if chroma_file_id else str(__import__('uuid').uuid4())
            )
            db.session.add(processed)
        db.session.commit()
        print(f"💾 Saved file info for user {user_id}: {filename} (file_id={chroma_file_id})")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error saving file info: {e}")