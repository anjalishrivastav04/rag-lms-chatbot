import os
import re
import hashlib
import redis
import fitz
import uuid
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from graph_handler import build_graph_from_chunks, delete_graph_for_file
from vision_handler import analyze_image, get_image_caption, classify_image_type

# 🚨 CRITICAL: Load environment variables BEFORE initializing any LangChain or local vision tools
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from ocr_handler import save_ocr_text_to_file, extract_text_from_image
from langchain_classic.indexes import SQLRecordManager, index
from pdf2image import convert_from_path

# --- SYSTEM INTEGRATION PATHS ---
# 🌟 Verified absolute path to your compiled OneDrive desktop binaries
POPPLER_PATH = r"C:\Users\iaman\OneDrive\Documents\Desktop\poppler-26.02.0\Library\bin"

# --- REDIS CONFIG ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# --- GROQ LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# --- EMBEDDINGS ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- RECORD MANAGER FOR INCREMENTAL INDEXING ---
record_manager = SQLRecordManager(
    namespace="rag-chatbot",
    db_url="sqlite:///record_manager_cache.db"
)
record_manager.create_schema()

# -------------------------------------------
# STATISTICAL ANALYSIS
# -------------------------------------------
def statistical_analysis(content):
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

# -------------------------------------------
# LLM FALLBACK
# -------------------------------------------
def ask_llm_for_chunking(sample_text):
    prompt = f"""Analyze this document sample and recommend the best text chunking strategy for a RAG system.
Choose ONLY ONE from: recursive, sentence, paragraph, sliding, structure, semantic

Document sample:
{sample_text}

Reply with ONLY the strategy name in lowercase, nothing else."""
    response = llm.invoke(prompt)
    method = response.content.strip().lower()
    valid = ["recursive", "sentence", "paragraph", "sliding", "structure", "semantic"]
    return method if method in valid else "recursive"

# -------------------------------------------
# CHUNKING METHOD DETECTOR (issue #5 — Redis -> Postgres -> statistical -> LLM)
# -------------------------------------------
def detect_chunking_method(content):
    file_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"chunking:{file_hash}"

    # 1. Check Redis FIRST — fast path, no DB hit
    cached = redis_client.get(cache_key)
    if cached:
        print(f"⚡ Chunking method from Redis cache: {cached}")
        return cached

    # 2. Redis missed — NOW check Postgres
    from models.models import ChunkingDecision
    from extensions import db
    existing = ChunkingDecision.query.get(file_hash)
    if existing:
        print(f"🗄️ Chunking method from Postgres cache: {existing.method}")
        redis_client.set(cache_key, existing.method)   # repopulate Redis
        return existing.method

    # 3. Both missed — compute fresh
    method, confidence = statistical_analysis(content)
    print(f"📊 Statistical result: {method} (confidence: {confidence:.0%})")

    if confidence < 0.75:
        print(f"🧠 Low confidence — asking LLM...")
        method = ask_llm_for_chunking(content[:2000])
        confidence = None  # LLM-picked method has no statistical confidence score
        print(f"🤖 LLM recommended: {method}")
    else:
        print(f"✅ High confidence — using: {method}")

    # 4. Write-through to both caches — Redis for speed, Postgres for durability
    redis_client.set(cache_key, method)
    print(f"💾 Chunking decision cached in Redis.")

    try:
        decision = ChunkingDecision(content_hash=file_hash, method=method, confidence=confidence)
        db.session.merge(decision)  # upsert — safe if this hash was already written concurrently
        db.session.commit()
        print(f"🗄️ Chunking decision persisted to Postgres.")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Failed to persist chunking decision to Postgres: {e}")

    return method

# -------------------------------------------
# EMBEDDED IMAGE OCR (issue #3 — hybrid PDF pages)
# -------------------------------------------
def extract_embedded_image_text(pdf_doc, page_num, main_text):
    """
    Finds embedded images on a given page (0-indexed) of an ALREADY-OPEN
    fitz document, runs OCR on each, and returns their text — skipping
    tiny/decorative images, low-confidence OCR noise, and text already
    substantially present in the page's main extracted text. Images that
    OCR can't sufficiently explain (diagrams, flowcharts) also get a
    VLM-generated caption appended, via vision_handler.get_image_caption().
    """
    ocr_blocks = []
    MIN_OCR_CONFIDENCE = 0.40  # confidence is a 0-1 fraction, not a percentage

    try:
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                if len(image_bytes) < 5000:
                    continue

                temp_img_path = f"documents/temp_embedded_{page_num + 1}_{img_index}.{image_ext}"
                with open(temp_img_path, "wb") as f:
                    f.write(image_bytes)

                try:
                    ocr_text, confidence = extract_text_from_image(temp_img_path)
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

                if not ocr_text or not ocr_text.strip():
                    continue

                # Skip low-confidence OCR — usually seals, watermarks,
                # or logos where OCR guesses garbage text.
                if confidence is not None and confidence < MIN_OCR_CONFIDENCE:
                    continue

                ocr_words = set(re.findall(r'\w+', ocr_text.lower()))
                main_words = set(re.findall(r'\w+', main_text.lower()))
                if ocr_words:
                    overlap = len(ocr_words & main_words) / len(ocr_words)
                    if overlap > 0.7:
                        continue

                # --- NEW: classify + optionally caption via VLM ---
                image_type = classify_image_type(ocr_text, confidence)

                if image_type == "diagram":
                    print(f"🖼️ Page {page_num + 1}, image {img_index}: classified as diagram — requesting VLM caption...")
                    temp_caption_path = f"documents/temp_caption_{page_num + 1}_{img_index}.{image_ext}"
                    with open(temp_caption_path, "wb") as f:
                        f.write(image_bytes)
                    try:
                        caption = get_image_caption(temp_caption_path)
                        if caption and caption.strip():
                            ocr_text = ocr_text.strip() + f"\n[IMAGE DESCRIPTION]: {caption.strip()}"
                    finally:
                        if os.path.exists(temp_caption_path):
                            os.remove(temp_caption_path)
                else:
                    print(f"📝 Page {page_num + 1}, image {img_index}: classified as text-dense — OCR sufficient, skipping VLM call.")
                # --- END NEW ---

                ocr_blocks.append(ocr_text.strip())

            except Exception as img_err:
                print(f"⚠️ Could not process embedded image {img_index} on page {page_num + 1}: {img_err}")
                continue
    except Exception as e:
        print(f"⚠️ Could not process embedded images for page {page_num + 1}: {e}")

    return ocr_blocks

# -------------------------------------------
# GET SPLITTER
# -------------------------------------------
def get_splitter(method):
    print(f"✂️ Applying chunking method: {method}")

    if method == "sentence":
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=[". ", "! ", "? ", "\n"]
        )
    elif method == "paragraph":
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". "]
        )
    elif method == "sliding":
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=[" ", "\n"]
        )
    elif method == "structure":
        return RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n## ", "\n# ", "\n### ", "\n\n", "\n"]
        )
    elif method == "semantic":
        return RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "]
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

# -------------------------------------------
# FILE VALIDATION (issue #2 — PDF only)
# -------------------------------------------
def check_document(file_name):
    if ".pdf" not in file_name:
        print(f"⚠️ Invalid file type: {file_name}")
        return False
    return True

# -------------------------------------------
# MAIN INGEST FUNCTION (issue #4 — single file, no Redis-driven duplicate detection)
# -------------------------------------------
def ingest_documents(file_name):
    if not check_document(file_name):
        return {}

    filepath = os.path.join("documents", file_name)
    if not os.path.exists(filepath):
        print(f"⚠️ File does not exist: {filepath}")
        return {}

    all_chunks = []
    file_chunk_counts = {}
    file_paths = [filepath]

    print(f"\n{'='*80}")
    print(f"🔍 INGESTION REPORT")
    print(f"{'='*80}")
    print(f"📊 Processing: {file_name}\n")

    for path in file_paths:
        filename = os.path.basename(path)

        file_id_key = f"file_id:{filename}"
        file_id = redis_client.get(file_id_key) or str(uuid.uuid4())
        redis_client.set(file_id_key, file_id)
        print(f"🔑 File ID for {filename}: {file_id}")

        print(f"📄 PROCESSING: {filename}")
        docs = []

        try:
            print(f"🔍 Loading PDF layout layers...")
            loader = PyPDFLoader(path)
            raw_pdf_docs = loader.load()

            fitz_doc = None
            try:
                fitz_doc = fitz.open(path)

                for page_num, doc in enumerate(raw_pdf_docs):
                    text_content = doc.page_content.strip()
                    clean_char_count = len(re.sub(r'\s+', '', text_content))

                    is_spaced_out = False
                    if len(text_content) > 0:
                        single_letters = len(re.findall(r'\b[a-zA-Z]\b', text_content))
                        total_words = max(len(text_content.split()), 1)
                        if (single_letters / total_words) > 0.40:
                            is_spaced_out = True

                    if clean_char_count > 30 and not is_spaced_out:
                        # Good selectable text — but may ALSO contain embedded
                        # images with their own text (screenshots, scanned
                        # signatures, diagrams, tables, etc.)
                        embedded_ocr_blocks = extract_embedded_image_text(fitz_doc, page_num, text_content)

                        if embedded_ocr_blocks:
                            print(f"🖼️ Page {page_num + 1}: found {len(embedded_ocr_blocks)} embedded image(s) with extractable text — merging.")
                            doc.page_content = text_content + "\n\n=== TEXT FROM EMBEDDED IMAGES ===\n" + "\n\n".join(embedded_ocr_blocks)
                            doc.metadata["has_embedded_image_text"] = True
                            doc.metadata["has_image_caption"] = "[IMAGE DESCRIPTION]" in doc.page_content

                        docs.append(doc)
                    else:
                        print(f"📸 Page {page_num + 1} flagged as image scan or broken layout. Initiating OCR fallback...")
                        try:
                            # 🌟 PASSED POPPLER BINARIES PATH DIRECTLY HERE:
                            images = convert_from_path(
                                path,
                                first_page=page_num + 1,
                                last_page=page_num + 1,
                                dpi=300,
                                poppler_path=POPPLER_PATH
                            )
                            if images:
                                temp_img_path = f"documents/temp_page_{page_num + 1}.jpg"
                                images[0].save(temp_img_path, 'JPEG')

                                ocr_text, confidence = extract_text_from_image(temp_img_path)

                                if ocr_text and ocr_text.strip():
                                    print(f"✨ Successfully extracted clean text from PDF Page {page_num + 1}!")
                                    docs.append(Document(
                                        page_content=ocr_text,
                                        metadata={
                                            "source": filename,
                                            "page": page_num + 1
                                        }
                                    ))
                                else:
                                    print(f"⚠️ OCR returned empty text for Page {page_num + 1}. Falling back to original (low-quality) text.")
                                    doc.metadata["degraded"] = True
                                    doc.metadata["degraded_reason"] = "ocr_empty"
                                    docs.append(doc)

                                if os.path.exists(temp_img_path):
                                    os.remove(temp_img_path)
                            else:
                                print(f"⚠️ No image rendered for Page {page_num + 1}. Falling back to original (low-quality) text.")
                                doc.metadata["degraded"] = True
                                doc.metadata["degraded_reason"] = "no_image_rendered"
                                docs.append(doc)
                        except Exception as pdf_img_err:
                            print(f"⚠️ Image-to-PDF parser skipped Page {page_num + 1}: {pdf_img_err}")
                            doc.metadata["degraded"] = True
                            doc.metadata["degraded_reason"] = f"ocr_exception: {pdf_img_err}"
                            docs.append(doc)
            finally:
                if fitz_doc is not None:
                    fitz_doc.close()

            if not docs:
                print(f"⚠️  No content extracted from {filename}\n")
                continue

            content = " ".join([d.page_content for d in docs])
            if not content.strip():
                print(f"⚠️  Empty content after processing {filename}\n")
                continue

            method = detect_chunking_method(content)
            splitter = get_splitter(method)
            chunks = splitter.split_documents(docs)

            # issue #6 — derive document title (embedded PDF title, else filename)
            raw_title = docs[0].metadata.get("title", "") if docs else ""
            document_title = raw_title.strip() if raw_title and raw_title.strip() else filename.rsplit('.', 1)[0]

            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename
                chunk.metadata["file_id"] = file_id
                chunk.metadata["filetype"] = filename.rsplit('.', 1)[1].lower()
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_method"] = method
                chunk.metadata["title"] = document_title       # issue #6
                chunk.metadata["category"] = "cybersecurity"   # issue #6
                chunk.metadata["status"] = "active"             # issue #1 + #6

            all_chunks.extend(chunks)
            file_chunk_counts[filename] = len(chunks)
            print(f"   ... ADDED: {len(chunks)} chunks\n")

            build_graph_from_chunks(chunks, filename)

            # ✅ Save source index to Redis for O(1) deletion later
            from redis_helpers import save_source_index
            chunk_ids = [chunk.metadata.get("chunk_index", i) for i, chunk in enumerate(chunks)]
            save_source_index(filename, chunk_ids)

        except Exception as e:
            print(f"❌ Failed processing {filename}: {str(e)}\n")
            continue

    if not all_chunks:
        print("⚠️ No new or updated text chunks gathered to index.")
        return file_chunk_counts

    # --- CHROMADB WORKSPACE INDEX HANDLING (issue #1) ---
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    print("💾 Connecting to ChromaDB server for synchronization...")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="rag_documents",
        embedding_function=embeddings,
    )
    print(f"🔀 Running parallel deduplication for {len(all_chunks)} incoming chunks...")

    try:
        sync_stats = index(
            all_chunks,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source"
        )
        print(f"📊 Synchronization Report: {sync_stats}")
        print(f"\n{'='*80}")
        print(f"✅ INGESTION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Added: {sync_stats.get('num_added', 0)} new chunks")
        print(f"⏭️  Skipped: {sync_stats.get('num_skipped', 0)} existing chunks")
        print(f"🗑️  Deleted: {sync_stats.get('num_deleted', 0)} old chunks")
        print(f"📊 Updated: {sync_stats.get('num_updated', 0)} chunks")
        print(f"{'='*80}\n")

    except ValueError as val_err:
        print(f"⚠️ Index Sync Desynchronization Detected: ({val_err})")

        if os.getenv("FORCE_REBUILD_ON_DESYNC", "false").lower() == "true":
            print("🚨 FORCE_REBUILD_ON_DESYNC is set — proceeding with full collection wipe + rebuild.")
            try:
                record_manager.delete_keys(record_manager.list_keys())
                print("🗑️ Cleared record manager tracking state (was at risk of desync).")
            except Exception as rm_err:
                print(f"⚠️ Could not clear record manager cleanly: {rm_err}")

            vectorstore.delete_collection()
            vectorstore = Chroma.from_documents(
                all_chunks, embeddings,
                client=chroma_client,
                collection_name="rag_documents",
            )   
            print("✅ Vector database re-indexed from this run's chunks only (FORCED).")
        else:
            print("❌ Ingestion for this file was NOT completed due to sync desync.")
            print("❌ No data was deleted. To force a full rebuild (WARNING: deletes ALL")
            print("❌ previously ingested documents), set FORCE_REBUILD_ON_DESYNC=true in .env and re-run.")
            return file_chunk_counts

    print("✅ ChromaDB collection updated successfully!")

    from kafka_handler import send_vectorstore_update
    send_vectorstore_update(file_name)

    return file_chunk_counts


if __name__ == "__main__":
    import sys
    from app import app  # reuse the existing Flask app instance for DB context

    with app.app_context():
        if len(sys.argv) > 1:
            ingest_documents(sys.argv[1])
        else:
            print("Usage: python ingest.py <filename.pdf>")