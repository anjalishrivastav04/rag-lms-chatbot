"""
clear_poisoned_cache.py
------------------------
Finds and removes cached Q&A entries that were poisoned by bad answers
(garbled OCR text, confused memory-narration responses, etc).

Two caches are cleaned:
  1. Redis exact-match cache   (key pattern: "rag:<question>")
  2. FAISS semantic cache      (on-disk store at CACHE_DIR)

Usage (run from project root, same folder as worker.py / app.py):
    python clear_poisoned_cache.py            # DRY RUN — only lists matches
    python clear_poisoned_cache.py --execute  # actually deletes matches
"""

import argparse
import os

from app import app
from extensions import redis_client, embeddings
from config import CACHE_DIR

# ---- known-bad keyword patterns --------------------------------------
# Add more substrings here any time you spot another poisoned entry.
BAD_KEYWORD_PATTERNS = [
    "favourite color",
    "favorite color",
    "tell me about the cv",
]


def matches_bad_pattern(text):
    text_lower = (text or "").lower()
    return any(p in text_lower for p in BAD_KEYWORD_PATTERNS)


def clean_redis_cache(execute):
    print("\n=== Redis exact-match cache (rag:*) ===")
    keys = list(redis_client.scan_iter(match="rag:*"))
    matched_keys = []
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        question_part = key_str[len("rag:"):]
        if matches_bad_pattern(question_part):
            matched_keys.append(key_str)
            print(f"  MATCH: {key_str}")

    print(f"Found {len(matched_keys)} matching Redis key(s) out of {len(keys)} total.")
    if execute and matched_keys:
        for k in matched_keys:
            redis_client.delete(k)
        print(f"✅ Deleted {len(matched_keys)} Redis key(s).")
    elif not execute and matched_keys:
        print("(dry run — nothing deleted. Re-run with --execute to delete.)")


def clean_semantic_cache(execute):
    from langchain_community.vectorstores import FAISS

    print("\n=== FAISS semantic cache ===")
    if not os.path.exists(CACHE_DIR):
        print("No semantic cache directory found — nothing to clean.")
        return

    cache_store = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
    docstore_dict = cache_store.docstore._dict  # {doc_id: Document}

    matched_ids = []
    for doc_id, doc in docstore_dict.items():
        question = doc.page_content
        response = doc.metadata.get("response", "")
        if matches_bad_pattern(question) or matches_bad_pattern(response):
            matched_ids.append(doc_id)
            print(f"  MATCH id={doc_id}")
            print(f"    Q: {question[:100]}")
            print(f"    A: {response[:150]}")

    print(f"\nFound {len(matched_ids)} matching semantic-cache entr"
          f"{'y' if len(matched_ids) == 1 else 'ies'} out of {len(docstore_dict)} total.")

    if not matched_ids:
        return

    if not execute:
        print("(dry run — nothing deleted. Re-run with --execute to delete.)")
        return

    try:
        cache_store.delete(matched_ids)
        cache_store.save_local(CACHE_DIR)
        print(f"✅ Deleted {len(matched_ids)} semantic-cache entries and re-saved index.")
    except Exception as e:
        # Some FAISS index types don't support in-place removal — rebuild instead.
        print(f"⚠️ In-place delete failed ({e}) — rebuilding index from remaining entries.")
        remaining_docs = [doc for doc_id, doc in docstore_dict.items() if doc_id not in matched_ids]
        if remaining_docs:
            new_store = FAISS.from_documents(remaining_docs, embeddings)
            new_store.save_local(CACHE_DIR)
        else:
            import shutil
            shutil.rmtree(CACHE_DIR)
        print(f"✅ Rebuilt semantic cache without {len(matched_ids)} bad entries.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete matches (default is dry-run/list-only)")
    args = parser.parse_args()

    with app.app_context():
        clean_redis_cache(args.execute)
        clean_semantic_cache(args.execute)

    if not args.execute:
        print("\n👉 This was a DRY RUN. Review the matches above, then run:")
        print("   python clear_poisoned_cache.py --execute")


if __name__ == "__main__":
    main()