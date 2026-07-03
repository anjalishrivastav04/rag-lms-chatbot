import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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