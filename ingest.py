import os
import re
import redis
import uuid
import hashlib
import chromadb
from dotenv import load_dotenv
from graph_handler import build_graph_from_chunks, delete_graph_for_file

# 🚨 CRITICAL: Load environment variables BEFORE initializing any LangChain or local vision tools
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from ocr_handler import save_ocr_text_to_file, extract_text_from_image
from langchain_classic.indexes import SQLRecordManager, index
from pdf2image import convert_from_path
import fitz  # PyMuPDF - for detecting & extracting embedded images per page
from vision_handler import analyze_image  # 👈 Safe to import now that environment is active!

# --- SYSTEM INTEGRATION PATHS ---
# 🌟 Verified absolute path to your compiled OneDrive desktop binaries
POPPLER_PATH = r"C:\Users\iaman\OneDrive\Documents\Desktop\poppler-26.02.0\Library\bin"

# --- REDIS CONFIG ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# --- GROQ LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

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
# CHUNKING METHOD DETECTOR
# -------------------------------------------
def detect_chunking_method(content):
    file_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"chunking:{file_hash}"

    cached = redis_client.get(cache_key)
    if cached:
        print(f"⚡ Chunking method from Redis cache: {cached}")
        return cached

    from models.models import ChunkingDecision
    from extensions import db
    existing = ChunkingDecision.query.get(file_hash)
    if existing:
        print(f"🗄️ Chunking method from Postgres cache: {existing.method}")
        redis_client.set(cache_key, existing.method)
        return existing.method

    method, confidence = statistical_analysis(content)
    print(f"📊 Statistical result: {method} (confidence: {confidence:.0%})")

    if confidence < 0.75:
        print(f"🧠 Low confidence — asking LLM...")
        method = ask_llm_for_chunking(content[:2000])
        confidence = None
        print(f"🤖 LLM recommended: {method}")
    else:
        print(f"✅ High confidence — using: {method}")

    redis_client.set(cache_key, method)
    print(f"💾 Chunking decision cached in Redis.")

    try:
        decision = ChunkingDecision(content_hash=file_hash, method=method, confidence=confidence)
        db.session.merge(decision)
        db.session.commit()
        print(f"🗄️ Chunking decision persisted to Postgres.")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Failed to persist chunking decision to Postgres: {e}")

    return method

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
# EMBEDDED IMAGE TEXT EXTRACTION
# -------------------------------------------
def extract_embedded_image_text(pdf_path, page_num, main_text):
    """
    Finds embedded images on a given PDF page (0-indexed). Some PDF exporters
    slice a single picture into many thin horizontal strips, so instead of
    OCR-ing each raw fragment separately (which reads only a sliver of text
    at a time), this clusters nearby/touching fragments into unified regions
    and renders each region as one high-res image before running OCR.
    Returns a list of OCR text blocks, roughly in top-to-bottom page order.
    """
    ocr_blocks = []
    try:
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)

        if not image_list:
            pdf_doc.close()
            return ocr_blocks

        # Collect the on-page bounding box of every embedded image
        all_rects = []
        for img_info in image_list:
            xref = img_info[0]
            for r in page.get_image_rects(xref):
                all_rects.append(r)

        if not all_rects:
            pdf_doc.close()
            return ocr_blocks

        # Cluster rects that are vertically adjacent/overlapping and share
        # roughly the same horizontal span into single unified regions —
        # this reconstructs images that got sliced into strips on export.
        all_rects = sorted(all_rects, key=lambda r: r.y0)
        clusters = []
        current = fitz.Rect(all_rects[0])
        gap_tolerance = 8  # points
        for r in all_rects[1:]:
            horizontally_overlaps = not (r.x1 < current.x0 or r.x0 > current.x1)
            vertically_close = (r.y0 - current.y1) <= gap_tolerance
            if horizontally_overlaps and vertically_close:
                current |= r
            else:
                clusters.append(current)
                current = fitz.Rect(r)
        clusters.append(current)

        for region_index, region_rect in enumerate(clusters):
            # Skip tiny regions (icons, bullets, decorative dividers)
            if region_rect.width < 20 or region_rect.height < 20:
                continue

            try:
                mat = fitz.Matrix(300 / 72, 300 / 72)  # render at 300 DPI
                pix = page.get_pixmap(matrix=mat, clip=region_rect)

                temp_img_path = f"documents/temp_embedded_{page_num + 1}_{region_index}.png"
                pix.save(temp_img_path)

                ocr_text, confidence = extract_text_from_image(temp_img_path)

                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

                if not ocr_text or not ocr_text.strip():
                    continue

                # Minimize duplication: skip if most of this OCR text
                # already appears in the page's main extracted text
                ocr_words = set(re.findall(r'\w+', ocr_text.lower()))
                main_words = set(re.findall(r'\w+', main_text.lower()))
                if ocr_words:
                    overlap = len(ocr_words & main_words) / len(ocr_words)
                    if overlap > 0.7:
                        continue

                ocr_blocks.append(ocr_text.strip())

            except Exception as region_err:
                print(f"⚠️ Could not process embedded image region {region_index} on page {page_num + 1}: {region_err}")
                continue

        pdf_doc.close()
    except Exception as e:
        print(f"⚠️ Could not open PDF for embedded image extraction: {e}")

    return ocr_blocks

# -------------------------------------------
# FILE VALIDATION
# -------------------------------------------
def check_document(file_name):
    if ".pdf" not in file_name:
        print(f"⚠️ Invalid file type: {file_name}")
        return False
    return True

# -------------------------------------------
# MAIN INGEST FUNCTION
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
    print(f"🔍 INCREMENTAL INGESTION REPORT")
    print(f"{'='*80}")
    print(f"📊 Total files to check: {len(file_paths)}\n")

    any_processed = False

    for path in file_paths:
        filename = os.path.basename(path)
        filename_lower = filename.lower()

        # Look up file_id from Redis (saved during upload)
        file_id_key = f"file_id:{filename}"
        file_id = redis_client.get(file_id_key) or str(uuid.uuid4())
        print(f"🔑 File ID for {filename}: {file_id}")

        if filename_lower.endswith('_ocr.txt') or filename_lower.endswith('_vision.txt'):
            print(f"⏭️  SKIPPED: {filename} (Temporary derivative file)\n")
            continue

        # -------------------------------------------------------------
        # 🧠 EARLY CACHE CHECK: Stop processing if the file is unchanged
        # -------------------------------------------------------------
        file_stat_sig = None
        try:
            stat = os.stat(path)
            file_stat_sig = f"{stat.st_mtime_ns}:{stat.st_size}"

            is_already_indexed = redis_client.get(f"fastpass_hash:{filename}")

            if is_already_indexed == file_stat_sig:
                print(f"⚡ FAST-PASS: {filename} has not changed. Skipping extraction and vision analysis entirely!\n")
                continue
        except Exception as hash_err:
            print(f"⚠️ Pre-computation skip check warning: {hash_err}")
        # -------------------------------------------------------------

        print(f"📄 PROCESSING: {filename}")
        any_processed = True
        docs = []

        try:
            # 1. Handle Text Files
            if filename_lower.endswith('.txt'):
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()

            # 2. Handle PDF Files (Smart Hybrid with Spacing Defect Detection)
            elif filename_lower.endswith('.pdf'):
                print(f"🔍 Loading PDF layout layers...")
                loader = PyPDFLoader(path)
                raw_pdf_docs = loader.load()

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
                        docs.append(doc)

                        embedded_ocr_blocks = extract_embedded_image_text(path, page_num, text_content)
                        if embedded_ocr_blocks:
                            print(f"🖼️ Page {page_num + 1}: found {len(embedded_ocr_blocks)} embedded image(s) with extractable text — merging.")
                            ocr_combined_text = "\n\n".join(embedded_ocr_blocks)
                            docs.append(Document(
                                page_content=f"[Content extracted from an embedded image/diagram on page {page_num + 1} of {filename}]\n{ocr_combined_text}",
                                metadata={
                                    "source": filename,
                                    "page": page_num + 1,
                                    "has_embedded_image_text": True,
                                    "is_embedded_image_chunk": True
                                }
                            ))
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

            # 3. Handle Pure Image Files with Toggle Configuration
            elif any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                vision_description = "Visual descriptions disabled via system config parameters."

                if os.getenv("USE_GOOGLE_VISION", "False").lower() == "true":
                    print(f"🖼️ Image detected - attempting analysis with local VLM engine...")
                    try:
                        vision_description = analyze_image(path)
                    except Exception as vision_err:
                        print(f"⚠️ Local VLM parsing failed. Bypassing structural description...")
                        vision_description = "Visual workspace descriptions unavailable. Falling back to clean OCR."
                else:
                    print(f"⚙️ Skipping VLM engine (USE_GOOGLE_VISION=False). Running local OCR framework pipeline...")

                print(f"📸 Running OCR for handwritten text...")
                output_txt_path, ocr_error = save_ocr_text_to_file(path)

                combined_text = f"=== IMAGE ANALYSIS SUMMARY ===\n{vision_description}\n\n"

                if output_txt_path and os.path.exists(output_txt_path):
                    with open(output_txt_path, 'r', encoding='utf-8') as f:
                        ocr_text = f.read()
                    combined_text += f"=== OCR EXTRACTED TEXT ===\n{ocr_text}"

                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(combined_text)

                    loader = TextLoader(output_txt_path, encoding="utf-8")
                    docs = loader.load()
                else:
                    temp_vision_path = f"documents/{os.path.splitext(filename)[0]}_vision.txt"
                    with open(temp_vision_path, 'w', encoding='utf-8') as f:
                        f.write(combined_text)
                    loader = TextLoader(temp_vision_path, encoding="utf-8")
                    docs = loader.load()

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

            # Derive a document title: prefer the PDF's own embedded title
            # metadata (set by the source app, e.g. "Print to PDF"), and
            # fall back to the filename (without extension) when a file
            # has no embedded title — e.g. .txt files, OCR'd images.
            raw_title = docs[0].metadata.get("title", "") if docs else ""
            document_title = raw_title.strip() if raw_title and raw_title.strip() else filename.rsplit('.', 1)[0]

            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename
                chunk.metadata["file_id"] = file_id
                chunk.metadata["filetype"] = filename.rsplit('.', 1)[1].lower()
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_method"] = method
                chunk.metadata["title"] = document_title
                chunk.metadata["category"] = "cybersecurity"
                chunk.metadata["status"] = "active"

            all_chunks.extend(chunks)
            print(f"   ... ADDED: {len(chunks)} chunks\n")
            file_chunk_counts[filename] = len(chunks)

            build_graph_from_chunks(chunks, filename)

            # ✅ Save source index to Redis for O(1) deletion later
            from redis_helpers import save_source_index
            chunk_ids = [chunk.metadata.get("chunk_index", i) for i, chunk in enumerate(chunks)]
            save_source_index(filename, chunk_ids)

            # Cache the successful hash state mapping inside Redis
            if file_stat_sig:
                redis_client.set(f"fastpass_hash:{filename}", file_stat_sig)

        except Exception as e:
            print(f"❌ Failed processing {filename}: {str(e)}\n")
            continue

    if not any_processed:
        print("⚡ All files verified via Fast-Pass checksum profiles. Vector space index is up to date!")
        return file_chunk_counts

    if not all_chunks:
        print("⚠️ No new or updated text chunks gathered to index.")
        return file_chunk_counts

    # --- CHROMADB WORKSPACE INDEX HANDLING ---
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
        print("🔄 Performing automatic index reconciliation rebuild...")

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
        print("✅ Vector database re-indexed from this run's chunks only.")
        print("⚠️ NOTE: This was a partial recovery — only chunks processed in this run are indexed.")
        print("⚠️ Files that were already indexed in prior runs and were NOT re-uploaded this time")
        print("⚠️ may be MISSING from the new index. Consider re-uploading all documents to be safe.")

    print("✅ ChromaDB collection updated successfully!")
    return file_chunk_counts


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_documents(sys.argv[1])
    else:
        print("Usage: python ingest.py <filename.pdf>")