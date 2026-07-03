import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from extensions import db, redis_client, embeddings, embedder
from models.models import SemanticCacheRecord
from config import CONTENT_TTL, DEFAULT_TTL, REDIS_TTL, CACHE_DIR, DISTANCE_THRESHOLD

# ============================================================
# --- REDIS CACHE ---
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

def save_to_redis_cache(question, response_text):
    try:
        cache_key = f"rag:{question.lower().strip()}"
        redis_client.setex(cache_key, REDIS_TTL, json.dumps(response_text))
    except Exception as e:
        print(f"⚠️ Redis write error: {e}")

# ============================================================
# --- SEMANTIC CACHE (PostgreSQL) ---
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
# --- SEMANTIC CACHE (FAISS) ---
# ============================================================

def check_semantic_cache(question):
    import os
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
    import os
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
# --- REDIS HELPERS ---
# ============================================================

def is_user_request_pending(user_id):
    lock_key = f"pending_request:{user_id}"
    return redis_client.exists(lock_key)

def set_user_request_pending(user_id, request_id, ttl=120):
    lock_key = f"pending_request:{user_id}"
    redis_client.setex(lock_key, ttl, request_id)

def clear_user_request_pending(user_id):
    lock_key = f"pending_request:{user_id}"
    redis_client.delete(lock_key)

def save_pending_parts(user_id, parts):
    try:
        redis_client.setex(f"pending_parts:{user_id}", 1800, json.dumps(parts))
    except Exception as e:
        print(f"⚠️ Save pending parts error: {e}")

def get_pending_parts(user_id):
    try:
        data = redis_client.get(f"pending_parts:{user_id}")
        return json.loads(data) if data else []
    except Exception as e:
        print(f"⚠️ Get pending parts error: {e}")
        return []

def clear_pending_parts(user_id):
    try:
        redis_client.delete(f"pending_parts:{user_id}")
    except Exception as e:
        print(f"⚠️ Clear pending parts error: {e}")

def get_chunk_ids_for_file(filename):
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
    try:
        base_name = filename.rsplit('.', 1)[0].lower()
        for key in [filename, f"{base_name}_ocr.txt", f"{base_name}_vision.txt"]:
            redis_client.delete(f"source_index:{key}")
        print(f"✅ Source index deleted for: {filename}")
    except Exception as e:
        print(f"⚠️ Source index delete error: {e}")

def add_to_blacklist(filename):
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
    try:
        return redis_client.smembers("deleted_files_blacklist")
    except:
        return set()

def check_rate_limit(user_id, token_count):
    from config import RATE_LIMIT_WINDOW, TOKEN_LIMIT_MAX
    try:
        key = f"rate_limit:{user_id}"
        is_new_key = not redis_client.exists(key)
        new_total = redis_client.incrby(key, token_count)
        if is_new_key:
            redis_client.expire(key, RATE_LIMIT_WINDOW)
        ttl = redis_client.ttl(key)
        reset_in = ttl if ttl > 0 else RATE_LIMIT_WINDOW
        if new_total > TOKEN_LIMIT_MAX:
            return False, 0, reset_in
        return True, TOKEN_LIMIT_MAX - new_total, reset_in
    except Exception as e:
        print(f"⚠️ Rate limit error: {e}")
        from config import TOKEN_LIMIT_MAX, RATE_LIMIT_WINDOW
        return True, TOKEN_LIMIT_MAX, RATE_LIMIT_WINDOW