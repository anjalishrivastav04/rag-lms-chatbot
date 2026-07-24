import re
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from extensions import db, llm, eval_llm, embeddings, embedder
from models.models import ProcessedFile
from services.intent_classifier import classify_intent
from services.cache import (
    check_redis_cache, save_to_redis_cache,
    check_semantic_cache, save_to_semantic_cache,
    cache_response_with_ttl, get_blacklist
)
from services.vectorstore import dense_retriever, bm25_retriever
from graph_handler import graph_retrieve
from config import RRF_K

logger = logging.getLogger("worker")

# ============================================================
# --- SAFE LLM INVOKE ---
# ============================================================

def safe_invoke(llm_instance, prompt, max_retries=2, wait_seconds=20):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return llm_instance.invoke(prompt)
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "413" in error_str or "429" in error_str:
                if attempt < max_retries:
                    logger.warning("llm_rate_limited_retrying", extra={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "wait_seconds": wait_seconds,
                    })
                    time.sleep(wait_seconds)
                    continue
            logger.error("llm_invoke_failed", exc_info=True, extra={"attempt": attempt + 1})
            raise
    raise last_error

# ============================================================
# --- RETRIEVAL ---
# ============================================================

def reciprocal_rank_fusion(list_of_doc_lists):
    """
    UPDATED: now accepts ANY number of document lists (was hardcoded to
    exactly bm25_docs + dense_docs before). This lets RAG Fusion reuse the
    same fusion math across many query-variation retrievals, not just the
    2 retrievers for a single query.
    """
    scores = {}
    doc_map = {}
    for doc_list in list_of_doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc
    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in ranked_keys]

def hybrid_retrieve(question):
    """Unchanged behavior — single-query BM25 + FAISS fusion. Still used
    by graph_answer.py / graph_tools.py and as a fallback."""
    from services.vectorstore import bm25_retriever, dense_retriever
    bm25_docs = bm25_retriever.invoke(question)
    dense_docs = dense_retriever.invoke(question)
    return reciprocal_rank_fusion([bm25_docs, dense_docs])

def retrieve_both(q):
    from services.vectorstore import bm25_retriever, dense_retriever
    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25_retriever.invoke, q)
        dense_future = executor.submit(dense_retriever.invoke, q)
        return bm25_future.result(), dense_future.result()
    
def generate_query_variations(question, n=3):
    """
    RAG Fusion step 1: generate n alternate phrasings of the question so
    retrieval has a better chance of matching documents that use different
    wording than the user's original question.
    """
    prompt = f"""Generate {n} different ways to phrase the following question,
so that a search engine has a better chance of finding relevant documents,
even if they use different wording than the original question.

Reply with ONLY a JSON array of strings, nothing else, no markdown fences.

QUESTION: {question}
"""
    try:
        response = safe_invoke(eval_llm, prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        variations = json.loads(content)
        if isinstance(variations, list) and variations:
            return [question] + [v for v in variations[:n] if isinstance(v, str)]
    except Exception:
        logger.warning("query_variation_generation_failed", exc_info=True)
    return [question]

def rag_fusion_retrieve(question, n_variations=3):
    variations = generate_query_variations(question, n=n_variations)
    from services.vectorstore import bm25_retriever, dense_retriever

    all_doc_lists = []
    for q in variations:
        bm25_docs, dense_docs = retrieve_both(q)
        all_doc_lists.append(bm25_docs)
        all_doc_lists.append(dense_docs)

    fused = reciprocal_rank_fusion(all_doc_lists)
    logger.info("rag_fusion_completed", extra={
        "original_question": question[:80],
        "num_variations": len(variations),
        "num_fused_docs": len(fused),
    })
    return fused

def is_academic_query(question):
    """Detects questions likely about course content (syllabus, curriculum,
    topics, etc.) rather than about a specific person. Used to decide
    whether resume/CV documents should be excluded before reranking."""
    academic_keywords = r"\b(syllabus|curriculum|course|module|semester|subject|topic|topics|lecture|class|chapter|unit|credits?|exam|assignment)\b"
    return bool(re.search(academic_keywords, question.lower()))

def is_person_document(source_filename):
    """Detects resume/CV-style documents by filename. These are frequently
    keyword-dense with tech terms (e.g. "Python, Flask, ...") which can
    fool the reranker into treating them as relevant to course-content
    questions when they're actually about a specific individual."""
    person_doc_patterns = r"(resume|_cv\b|^cv\.|curriculum[_\s]?vitae)"
    return bool(re.search(person_doc_patterns, source_filename.lower()))

def filter_person_docs_for_academic_query(question, docs):
    """If the question looks academic (about course content, not a
    specific person), drop resume/CV documents before reranking so they
    can't crowd out genuinely relevant course material."""
    if not is_academic_query(question):
        return docs
    filtered = [d for d in docs if not is_person_document(d.metadata.get("source", ""))]
    removed = len(docs) - len(filtered)
    if removed > 0:
        logger.info("person_docs_filtered_for_academic_query", extra={
            "question": question[:80],
            "removed_count": removed,
        })
    return filtered if filtered else docs  # never filter down to nothing


def rerank_documents(question, docs, top_n=6, max_per_source=3, min_score=0.05):
    """
    UPDATED: added min_score threshold. Previously, if max_per_source
    caused genuinely relevant chunks to be skipped, the function would
    backfill top_n using leftover docs regardless of their score — even
    if that score was 0.0 (completely irrelevant). Now, backfill only
    happens with docs scoring above min_score. If fewer than top_n
    qualify, we return fewer docs rather than padding with garbage.
    """
    from extensions import reranker
    from flashrank import RerankRequest
    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
        return docs[:top_n]

    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = reranker.rerank(rerank_request)
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    selected_indices = []
    source_counts = {}
    leftover_indices = []

    for r in ranked:
        idx = r["id"]
        score = r["score"]
        source = docs[idx].metadata.get("source", "unknown")
        if source_counts.get(source, 0) < max_per_source:
            selected_indices.append(idx)
            source_counts[source] = source_counts.get(source, 0) + 1
        elif score >= min_score:
            leftover_indices.append(idx)
        # else: score too low AND source already capped -> drop entirely
        if len(selected_indices) >= top_n:
            break

    if len(selected_indices) < top_n:
        selected_indices.extend(leftover_indices[: top_n - len(selected_indices)])

    logger.info("rerank_completed", extra={
        "question": question[:80],
        "num_candidates": len(docs),
        "num_selected": len(selected_indices),
        "distinct_sources": len(source_counts),
    })

    return [docs[i] for i in selected_indices]
# ============================================================
# --- QUERY HELPERS ---
# ============================================================
def is_personal_statement(question):
    intent, confidence = classify_intent(question)
    if intent == "personal_statement" and confidence >= 0.60:
        return True

    # Fallback to original regex if classifier isn't confident
    personal_patterns = r"\bi\s+(?:actually|also|just|now|really)?\s*(prefer|like|love|hate|am\b|'m\b|work at|live in|study|was born|have a|own a)\b|\bmy\s+(favorite|favourite)\b|\bmy\s+name\s+is\b"
    return bool(re.search(personal_patterns, question.lower().strip()))

def is_confirmation_or_statement(question):
    """Detects simple agreement/confirmation/feedback statements that are
    NOT questions (e.g. 'that's correct', 'yes she was there too'). These
    should never go through decompose_question's LLM call, which tends to
    mis-split them into fake 'sub-questions' since there's no real
    question content to decompose.

    NARROWED: previously any message without a question mark AND without
    one of a small set of interrogative words was treated as a statement.
    This caused real questions/requests phrased without those exact words
    (e.g. "give me details about X", "summarize the syllabus") to be
    wrongly routed as casual chit-chat, skipping retrieval entirely. Now
    the catch-all only fires for genuinely short phrases (<=4 words), and
    the interrogative list includes common imperative request-verbs too
    (summarize/give/share/provide/...), so short real asks aren't
    misclassified either. Longer messages default to NOT being a
    statement — better to risk a mis-skipped decomposition than to
    silently bypass retrieval for a real question.
    """
    q = question.lower().strip()

    confirmation_starts = r"^(yes|yeah|yup|nope|no|correct|exactly|agreed|i agree|that'?s (correct|right|true|wrong|not right)|thats (correct|right|true|wrong)|sounds (right|good)|makes sense)\b"
    if re.search(confirmation_starts, q):
        return True
    
    implicit_question_starts = r"^(tell|explain|describe|show|give|from|about|regarding|summarize|summarise|elaborate|discuss|outline|detail|provide|share)"
    if re.search(implicit_question_starts, q):
        return False
    
    word_count = len(q.split())
    if word_count <= 4:
        interrogative_words = r"\b(what|who|when|where|why|how|which|whom|whose|can|could|do|does|did|is|are|will|would|should|explain|tell me|list|describe|summarize|summarise|give|share|provide|elaborate|discuss|outline|show|tell|detail|details)\b"
        has_question_mark = "?" in question
        has_interrogative = bool(re.search(interrogative_words, q))
        if not has_question_mark and not has_interrogative:
            return True

    return False

def is_casual_query(question):
    intent, confidence = classify_intent(question)
    if intent == "greeting" and confidence >= 0.60:
        return True

    # Fallback to original regex logic if classifier isn't confident
    q = question.lower().strip()
    if len(q) <= 2 and not q.isdigit():
        return True
    if is_personal_statement(question):
        return True
    if is_confirmation_or_statement(question):
        return True
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|wassup|yo|who are you|what is your name|how are you|what can you do|help|thanks|thank you|okay|ok|cool|nice|great|bye|goodbye|what do you mean|clarify|rephrase|tell more|i prefer|more detail)\b"
    if re.search(greetings, q):
        return True
    if re.match(r'^\d+\.?\s*$', q):
        return True
    return False

def is_list_documents_query(question):
    # (?<!\.) prevents matching "pdf"/"file"/"document" when it's actually
    # part of a filename extension like "test_diagram_only_v2.pdf" —
    # previously any question containing "what"/"which" ANYWHERE plus a
    # .pdf filename anywhere later would wrongly match, since \b matches
    # right after the dot in ".pdf" too. Now requires the word "pdf" etc.
    # to NOT be immediately preceded by a period.
    patterns = r"\b(list|show|what|which)\b.*(?<!\.)\b(document|file|pdf)s?\b"
    return bool(re.search(patterns, question.lower().strip()))

def decompose_question(question):
    if is_casual_query(question) or is_list_documents_query(question):
        return [{"label": question, "question": question}]
    
    # ✅ Only bypass decomposition, NOT retrieval
    # These patterns are clearly single questions — skip LLM decomposition call
    # but still go through normal RAG pipeline
    single_question_patterns = r"^(tell me about|explain more|describe more|give me more|provide more)"
    if re.match(single_question_patterns, question.lower().strip()):
        return [{"label": question, "question": question}]
    
    # rest of decompose logic...

    decompose_prompt = f"""You are a strict question analyzer.

ONLY split a question if it EXPLICITLY asks about 2 or more COMPLETELY DIFFERENT topics joined by "and", "also", "as well as", or similar connectors.

IMPORTANT RULES:
- The "question" field must ALWAYS be a full standalone sentence, NEVER a single word or phrase
- The "label" is just a short 3-5 word summary tag
- Synonyms like "students or members", "files or documents" count as ONE topic — do NOT split them
- If in doubt, return as SINGLE

Examples that should NOT be split:
- "who are the students or members working under saurabh sir" → SINGLE
- "can you tell about the ongoing capstone project" → SINGLE
- "who are the members of the project under saurabh sir" → SINGLE
- "what are we doing in the capstone project" → SINGLE
- "explain more about the punjabi dialect project" → SINGLE
- "tell me about X" → SINGLE

Examples that SHOULD be split:
- "what is X and how does Y work" → MULTIPLE (two different topics)
- "tell me about X and also about Y" → MULTIPLE (two different subjects)

If single, return the question unchanged as one complete sentence.

Reply in this EXACT JSON format only, nothing else, no markdown:
{{"parts": [{{"label": "short 3-5 word label", "question": "full standalone sub-question as a complete sentence"}}]}}

QUESTION: {question}
"""
    try:
        response = safe_invoke(eval_llm, decompose_prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        parts = data.get("parts", [])
        if isinstance(parts, list) and len(parts) > 1:
            return [{"label": p["label"], "question": p["question"]} for p in parts if p.get("question")]
        return [{"label": question, "question": question}]
    except Exception:
        logger.warning("question_decomposition_failed", exc_info=True)
        return [{"label": question, "question": question}]

def generate_followup_options(question, answer):
    try:
        options_prompt = f"""A user asked this specific question and got this answer.
Generate exactly 3 short follow-up questions that are DIRECTLY related to THIS specific question and answer only.

QUESTION: {question}
ANSWER: {answer[:800]}

Rules:
- Each option must be under 8 words
- Must be DIRECTLY about the same topic as the question above
- Do NOT generate generic questions about unrelated topics
- Make them feel like a natural next question a curious person would ask
- Return ONLY a JSON array, nothing else
- Example for a question about Django: ["How does Django handle migrations?", "What is Django ORM?", "How to deploy Django apps?"]

JSON array:"""

        options_response = eval_llm.invoke(options_prompt)
        content = options_response.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        options = json.loads(content)
        return options[:4] if isinstance(options, list) else []
    except Exception:
        logger.warning("options_generation_failed", exc_info=True)
        return []

def format_multi_part_reply(parts):
    sections = []
    for i, part in enumerate(parts, start=1):
        section = f"**{i}. {part['question']}**\n{part['answer']}"
        if part['low_confidence']:
            section += "\n\n⚠️ *I'm not fully confident in this part of the answer — it may be incorrect.*"
        sections.append(section)
    return "\n\n".join(sections)

# ============================================================
# --- DB HELPERS ---
# ============================================================

def get_past_chat_history(session_id, limit=6):
    from models.models import ChatHistory
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
    except Exception:
        logger.error("chat_history_fetch_failed", exc_info=True, extra={"session_id": session_id})
        return "No previous history.\n"

def save_chat_message(user_id, session_id, role, content, cache_source='NONE', response_time_ms=0, ip_address=None):
    from models.models import ChatHistory
    try:
        msg = ChatHistory(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            cache_source=cache_source,
            response_time_ms=response_time_ms,
            ip_address=ip_address
        )
        db.session.add(msg)
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.error("chat_message_save_failed", exc_info=True, extra={
            "user_id": user_id,
            "session_id": session_id,
        })

def ingest_feedback_to_vectorstore(question, correct_answer, feedback_id):
    import services.vectorstore as vs_module
    try:
        content = f"Q: {question}\nA: {correct_answer}"
        doc = Document(
            page_content=content,
            metadata={
                "source": f"admin_feedback_{feedback_id}",
                "filetype": "feedback",
                "chunk_index": 0,
                "question": question,
                "is_admin_feedback": True,
                "status": "active"
            }
        )
        if vs_module.vectorstore:
            vs_module.vectorstore.add_documents([doc])
            vs_module.ALL_DOCS.append(doc)
            vs_module.bm25_retriever = BM25Retriever.from_documents(vs_module.ALL_DOCS, k=6)
            cache_response_with_ttl(
                user_id=1,
                query_text=question,
                query_embedding=embedder.encode(question),
                response=correct_answer,
                content_type='admin_feedback'
            )
            logger.info("feedback_ingested", extra={"feedback_id": feedback_id, "question": question[:50]})
            return True
        else:
            logger.error("vectorstore_not_initialized", extra={"feedback_id": feedback_id})
            return False
    except Exception:
        logger.error("feedback_ingestion_failed", exc_info=True, extra={"feedback_id": feedback_id})
        return False

# ============================================================
# --- MAIN ANSWER FUNCTION ---
# ============================================================

def get_answer(question, session_id):
    past_history = get_past_chat_history(session_id, limit=6)
    blacklist = get_blacklist()
    blacklist_str = ", ".join(blacklist) if blacklist else "None"

    if is_casual_query(question):
        logger.info("casual_query_routed", extra={"session_id": session_id})
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
        response = safe_invoke(llm, prompt)
        return response.content, "NONE", "⚡ Direct LLM Conversation (No Document Search)", []

    if is_list_documents_query(question):
        try:
            blacklist_filenames = set(get_blacklist())
            files = ProcessedFile.query.all()
            valid_files = [f for f in files if f.filename not in blacklist_filenames]
            if not valid_files:
                return "There are no documents available right now.", "NONE", "📋 Direct database lookup", []
            file_list = "\n".join([f"- {f.file_id}" for f in valid_files])
            answer = f"Here are the available documents:\n\n{file_list}"
            return answer, "NONE", "📋 Direct database lookup (no LLM hallucination)", []
        except Exception:
            logger.warning("list_documents_query_failed", exc_info=True)

    cached = check_redis_cache(question)
    if cached:
        logger.info("cache_hit", extra={"cache_source": "REDIS", "session_id": session_id})
        return cached, "REDIS", "", []

    cached = check_semantic_cache(question)
    if cached:
        save_to_redis_cache(question, cached)
        logger.info("cache_hit", extra={"cache_source": "SEMANTIC", "session_id": session_id})
        return cached, "SEMANTIC", "", []

    retrieval_start = time.time()
    docs = rag_fusion_retrieve(question)   # CHANGED: was hybrid_retrieve(question)
    docs = filter_person_docs_for_academic_query(question, docs)  # NEW: drop resume/CV docs for academic questions
    docs = rerank_documents(question, docs, top_n=6)
    retrieval_latency_ms = round((time.time() - retrieval_start) * 1000, 2)
    logger.info("retrieval_completed", extra={
        "session_id": session_id,
        "latency_ms": retrieval_latency_ms,
        "num_docs": len(docs),
    })

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
            logger.info("blacklist_filter_applied", extra={"chunks_removed": filtered_count})

    graph_context, related_entities = graph_retrieve(question)
    extra_docs = []
    if related_entities:
        from services.vectorstore import dense_retriever
        for entity in related_entities[:5]:
            entity_docs = dense_retriever.invoke(entity)
            entity_docs = [
                d for d in entity_docs
                if not any(
                    bl.lower().rsplit('.', 1)[0] in d.metadata.get("source", "").lower()
                    for bl in blacklist
                )
            ]
            extra_docs.extend(entity_docs)
        existing_contents = {doc.page_content for doc in docs}
        MAX_TOTAL_DOCS = 10
        for doc in extra_docs:
            if len(docs) >= MAX_TOTAL_DOCS:
                break
            if doc.page_content not in existing_contents:
                docs.append(doc)
                existing_contents.add(doc.page_content)
        logger.info("graph_entities_added", extra={
            "num_entities": len(related_entities[:5]),
            "total_docs": len(docs),
        })

    if not docs or (len(docs) == 1 and docs[0].metadata.get("source") == "system"):
    # ✅ Fallback to LLM general knowledge
       print("💡 No docs found — falling back to LLM general knowledge")
       general_prompt = f"""You are a helpful AI assistant with broad general knowledge.
    The user asked a question that is not covered in the uploaded documents.
    Answer it using your general knowledge clearly and concisely.
    At the end, mention that this answer is from general knowledge, not from uploaded documents.

    ### PAST CONVERSATION:
    {past_history}

    ### QUESTION:
    {question}

    Answer:"""
       response = safe_invoke(llm, general_prompt)
       answer = response.content
       options = generate_followup_options(question, answer)
       return answer, "GENERAL_LLM", "💡 Answered from General Knowledge (not from uploaded documents)", options

    context = ""
    sources_set = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filetype = doc.metadata.get("filetype", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        sources_set.add(source)
        file_record = ProcessedFile.query.filter_by(filename=source).first()
        file_id = file_record.file_id if file_record else "UNKNOWN"
        context += f"[FileID: {file_id} | Type: {filetype} | Chunk: {chunk_index}]\n"
        context += doc.page_content[:600] + "\n\n"

    alpha = 0.7
    retrieval_info = f"📊 Retrieved using: RAG Fusion (multi-query) + {int(alpha*100)}% Semantic (Chroma) + {int((1-alpha)*100)}% Keyword (BM25)\n📎 Sources: {', '.join(sources_set)}"

    prompt = f"""You are an expert, smart, helpful document assistant. Your job is to answer questions based strictly on the provided context.

STRICT RULES — follow these without exception:
1. Answer in clean, natural, conversational language only.
2. NEVER display source tags like [Source:...], [Type:...], or [Chunk:...] in your answer.
3. NEVER reveal actual filenames to the user — always refer to documents by their File ID only.
4. When asked what files/documents are available, list only File IDs that appear EXACTLY as written in the context below.
5. If the answer is found in the context, answer confidently and completely.
6. If multiple documents have relevant info, combine and summarize from ALL of them.
7. If the answer is NOT in the context, say clearly: "I don't have information about this in the uploaded documents."
8. NEVER make up or guess information that is not in the context.
9. If the document mentions a year or date, use that — do NOT use today's date or assume the current year is 2026.
10. The following files have been DELETED — completely IGNORE any information from them: {blacklist_str}
11. Keep your answer focused and concise — no unnecessary filler or repetition.
12. NEVER mention File IDs, document identifiers, or phrases like "the document with FileID..." or "discussed in FileID..." anywhere in your answer. Simply state the information directly and naturally, as if you already know it — do not cite or reference which file/chunk it came from in any way. Source attribution is handled separately by the system, not by you.
### PAST CONVERSATION:
{past_history}

### KNOWLEDGE GRAPH:
{graph_context}

### CONTEXT:
{context}

### QUESTION:
{question}

Answer:"""

    llm_start = time.time()
    response = safe_invoke(llm, prompt)
    llm_latency_ms = round((time.time() - llm_start) * 1000, 2)
    answer = response.content
    logger.info("llm_answer_generated", extra={
        "session_id": session_id,
        "latency_ms": llm_latency_ms,
        "num_sources": len(sources_set),
    })

    options = generate_followup_options(question, answer)
    if "[Source:" not in answer and "[Type:" not in answer and "[FileID:" not in answer:
        save_to_redis_cache(question, answer)
        save_to_semantic_cache(question, answer)
    else:
        logger.warning("bad_response_detected_skipping_cache", extra={"session_id": session_id})

    return answer, "NONE", retrieval_info, options



