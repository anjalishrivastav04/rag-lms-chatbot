"""
ingest/chunking.py
------------------
Smart chunking strategy: statistical analysis → optional LLM fallback
→ Redis/Postgres caching → splitter factory.
"""

import re
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Helpers ────────────────────────────────────────────────────────────────────

def statistical_analysis(content: str):
    """
    Analyse document content and return (method, confidence) based on
    structural signals (headers, code blocks, tables, math, bullets …).
    """
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    words = content.split()
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 20]

    if not sentences:
        return "recursive", 0.50

    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    vocabulary_richness = len(set(words)) / max(len(words), 1)
    paragraph_count = len(paragraphs)
    has_headers = bool(re.search(r'\n#{1,3} |\n[A-Z][^\n]{0,50}\n[-=]+', content))
    has_tables = bool(re.search(r'\|.*\|.*\|', content))
    has_code = bool(re.search(r'```|def |class |import |function ', content))
    has_math = bool(re.search(r'\$.*\$|equation|formula|theorem', content, re.IGNORECASE))
    has_bullets = bool(re.search(r'\n[-*•] ', content))
    doc_length = len(content)

    print(f"📊 Document Stats:")
    print(f"   Avg sentence length: {avg_sentence_len:.1f} words")
    print(f"   Vocabulary richness: {vocabulary_richness:.2f}")
    print(f"   Paragraphs: {paragraph_count}")
    print(f"   Headers: {has_headers} | Tables: {has_tables} | Code: {has_code} | Math: {has_math}")

    if has_code and has_headers:
        return "structure", 0.88
    if has_headers and paragraph_count > 3:
        return "recursive", 0.90
    if has_math and avg_sentence_len > 20:
        return "semantic", 0.72
    if has_tables and not has_code:
        return "structure", 0.85
    if paragraph_count > 10 and avg_sentence_len > 15:
        return "paragraph", 0.85
    if avg_sentence_len < 12 and has_bullets:
        return "sentence", 0.82
    if doc_length > 10000 and paragraph_count < 5:
        return "sliding", 0.80
    return "recursive", 0.70


def ask_llm_for_chunking(sample_text: str) -> str:
    """Ask the LLM to recommend a chunking strategy when statistical confidence is low."""
    from llm_provider import create_chat_model
    llm = create_chat_model(temperature=0.0)
    prompt = f"""Analyze this document sample and recommend the best text chunking strategy for a RAG system.
Choose ONLY ONE from: recursive, sentence, paragraph, sliding, structure, semantic

Document sample:
{sample_text}

Reply with ONLY the strategy name in lowercase, nothing else."""
    response = llm.invoke(prompt)
    method = response.content.strip().lower()
    valid = ["recursive", "sentence", "paragraph", "sliding", "structure", "semantic"]
    return method if method in valid else "recursive"


# ── Main class ─────────────────────────────────────────────────────────────────

class ChunkingStrategy:
    """
    Determines the best chunking method for a document using a two-stage
    pipeline: fast statistical analysis → LLM fallback when confidence < 0.75.
    Results are cached in Redis (fast) and Postgres (persistent).
    """

    def __init__(self, redis_client):
        self._redis = redis_client

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, content: str) -> str:
        """Return the best chunking method name for this content."""
        file_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"chunking:{file_hash}"

        # 1. Redis cache
        cached = self._redis.get(cache_key)
        if cached:
            print(f"⚡ Chunking method from Redis cache: {cached}")
            return cached

        # 2. Postgres cache
        method = self._check_postgres(file_hash)
        if method:
            self._redis.set(cache_key, method)
            return method

        # 3. Statistical analysis
        method, confidence = statistical_analysis(content)
        print(f"📊 Statistical result: {method} (confidence: {confidence:.0%})")

        if confidence < 0.75:
            print("🧠 Low confidence — asking LLM...")
            method = ask_llm_for_chunking(content[:2000])
            confidence = None
            print(f"🤖 LLM recommended: {method}")
        else:
            print(f"✅ High confidence — using: {method}")

        # Persist
        self._redis.set(cache_key, method)
        print("💾 Chunking decision cached in Redis.")
        self._persist_postgres(file_hash, method, confidence)

        return method

    def get_splitter(self, method: str) -> RecursiveCharacterTextSplitter:
        """Return the appropriate LangChain text splitter for the given method."""
        print(f"✂️ Applying chunking method: {method}")
        configs = {
            "sentence":  dict(chunk_size=200,  chunk_overlap=20,  separators=[". ", "! ", "? ", "\n"]),
            "paragraph": dict(chunk_size=800,  chunk_overlap=100, separators=["\n\n", "\n", ". "]),
            "sliding":   dict(chunk_size=400,  chunk_overlap=100, separators=[" ", "\n"]),
            "structure": dict(chunk_size=600,  chunk_overlap=50,  separators=["\n## ", "\n# ", "\n### ", "\n\n", "\n"]),
            "semantic":  dict(chunk_size=600,  chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]),
        }
        kwargs = configs.get(method, dict(chunk_size=500, chunk_overlap=50))
        return RecursiveCharacterTextSplitter(**kwargs)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _check_postgres(self, file_hash: str):
        try:
            from models.models import ChunkingDecision
            existing = ChunkingDecision.query.get(file_hash)
            if existing:
                print(f"🗄️ Chunking method from Postgres cache: {existing.method}")
                return existing.method
        except Exception:
            pass
        return None

    def _persist_postgres(self, file_hash: str, method: str, confidence):
        try:
            from models.models import ChunkingDecision
            from extensions import db
            decision = ChunkingDecision(content_hash=file_hash, method=method, confidence=confidence)
            db.session.merge(decision)
            db.session.commit()
            print("🗄️ Chunking decision persisted to Postgres.")
        except Exception as e:
            try:
                from extensions import db
                db.session.rollback()
            except Exception:
                pass
            print(f"⚠️ Failed to persist chunking decision to Postgres: {e}")
