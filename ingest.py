import os
import re
import glob
import redis
import hashlib
import fitz  # PyMuPDF - for detecting & extracting embedded images per page
import uuid
from dotenv import load_dotenv
from graph_handler import build_graph_from_chunks, delete_graph_for_file

# 🚨 CRITICAL: Load environment variables BEFORE initializing any LangChain or local vision tools
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from ocr_handler import save_ocr_text_to_file, extract_text_from_image  
from langchain_classic.indexes import SQLRecordManager, index
from pdf2image import convert_from_path  
from vision_handler import analyze_image  # 👈 Safe to import now that environment is active!

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
# CHUNKING METHOD DETECTOR
# -------------------------------------------
def detect_chunking_method(content):
    file_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"chunking:{file_hash}"

    cached = redis_client.get(cache_key)
    if cached:
        print(f"⚡ Chunking method from Redis cache: {cached}")
        return cached

    method, confidence = statistical_analysis(content)
    print(f"📊 Statistical result: {method} (confidence: {confidence:.0%})")

    if confidence < 0.75:
        print(f"🧠 Low confidence — asking LLM...")
        method = ask_llm_for_chunking(content[:2000])
        print(f"🤖 LLM recommended: {method}")
    else:
        print(f"✅ High confidence — using: {method}")

    redis_client.set(cache_key, method)
    print(f"💾 Chunking decision cached in Redis.")
    return method

# -------------------------------------------
# GET SPLITTER
# -------------------------------------------
def extract_embedded_image_text(pdf_doc, page_num, main_text):
    """
    Finds embedded images on a given page (0-indexed) of an ALREADY-OPEN
    fitz document, runs OCR on each, and returns their text — skipping
    tiny/decorative images, low-confidence OCR noise, and text already
    substantially present in the page's main extracted text.
    """
    ocr_blocks = []
    MIN_OCR_CONFIDENCE = 0.40  # tune this based on real-world results

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

                # ✅ Skip low-confidence OCR — usually seals, watermarks,
                # or logos where OCR guesses garbage text rather than
                # extracting real content.
                if confidence is not None and confidence < MIN_OCR_CONFIDENCE:
                    continue

                ocr_words = set(re.findall(r'\w+', ocr_text.lower()))
                main_words = set(re.findall(r'\w+', main_text.lower()))
                if ocr_words:
                    overlap = len(ocr_words & main_words) / len(ocr_words)
                    if overlap > 0.7:
                        continue

                ocr_blocks.append(ocr_text.strip())

            except Exception as img_err:
                print(f"⚠️ Could not process embedded image {img_index} on page {page_num + 1}: {img_err}")
                continue

    except Exception as e:
        print(f"⚠️ Could not process embedded images for page {page_num + 1}: {e}")

    return ocr_blocks

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
# MAIN INGEST FUNCTION
# -------------------------------------------
def ingest_documents():
    all_chunks = []

    file_paths = (
        glob.glob("documents/*.txt") + 
        glob.glob("documents/*.pdf") + 
        glob.glob("documents/*.jpg") + 
        glob.glob("documents/*.jpeg") + 
        glob.glob("documents/*.png") + 
        glob.glob("documents/*.bmp") + 
        glob.glob("documents/*.gif")
    )

    if not file_paths:
        print("📁 No documents found in /documents folder.")
        return

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
        # ✅ END ADD
        
        if filename_lower.endswith('_ocr.txt') or filename_lower.endswith('_vision.txt'):
            print(f"⏭️  SKIPPED: {filename} (Temporary derivative file)\n")
            continue
            
        # -------------------------------------------------------------
        # 🧠 EARLY CACHE CHECK: Stop processing if the file is unchanged
        # -------------------------------------------------------------
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
                # ✅ Good selectable text — but may ALSO contain embedded
                # images with their own text (screenshots, scanned
                # signatures, diagrams, tables, etc.)
                            embedded_ocr_blocks = extract_embedded_image_text(fitz_doc, page_num, text_content)

                            if embedded_ocr_blocks:     
                               print(f"🖼️ Page {page_num + 1}: found {len(embedded_ocr_blocks)} embedded image(s) with extractable text — merging.")
                               doc.page_content = text_content + "\n\n=== TEXT FROM EMBEDDED IMAGES ===\n" + "\n\n".join(embedded_ocr_blocks)
                               doc.metadata["has_embedded_image_text"] = True

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

            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename 
                chunk.metadata["file_id"] = file_id 
                chunk.metadata["filetype"] = filename.rsplit('.', 1)[1].lower()
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_method"] = method

            all_chunks.extend(chunks)
            print(f"   ... ADDED: {len(chunks)} chunks\n")

            build_graph_from_chunks(chunks, filename)

            # ✅ Save source index to Redis for O(1) deletion later
            from redis_helpers import save_source_index
            chunk_ids = [chunk.metadata.get("chunk_index", i) for i, chunk in enumerate(chunks)]
            save_source_index(filename, chunk_ids)
            
            # Cache the successful hash state mapping inside Redis
            redis_client.set(f"fastpass_hash:{filename}", file_stat_sig)
            
        except Exception as e:
            print(f"❌ Failed processing {filename}: {str(e)}\n")
            continue

    if not any_processed:
        print("⚡ All files verified via Fast-Pass checksum profiles. Vector space index is up to date!")
        return

    if not all_chunks:
        print("⚠️ No new or updated text chunks gathered to index.")
        return

    # --- STABILIZED FAISS WORKSPACE INDEX HANDLING ---
    faiss_dir = "vectorstore"
    faiss_file = os.path.join(faiss_dir, "index.faiss")
    
    if os.path.exists(faiss_dir) and os.path.exists(faiss_file):
        print("💾 Loading existing FAISS index for synchronization...")
        vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("✨ Building fresh vector footprint pipeline...")
        first_chunk = all_chunks.pop(0)
        vectorstore = FAISS.from_documents([first_chunk], embeddings)

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
        
        # Clear record_manager's tracking state so it doesn't reference
        # chunks/sources that may no longer match the rebuilt FAISS index.
        try:
            record_manager.delete_keys(record_manager.list_keys())
            print("🗑️ Cleared record manager tracking state (was at risk of desync).")
        except Exception as rm_err:
            print(f"⚠️ Could not clear record manager cleanly: {rm_err}")
        
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        print("✅ Vector database re-indexed from this run's chunks only.")
        print("⚠️ NOTE: This was a partial recovery — only chunks processed in this run are indexed.")
        print("⚠️ Files that were already indexed in prior runs and were NOT re-uploaded this time")
        print("⚠️ may be MISSING from the new index. Consider re-uploading all documents to be safe.")

    vectorstore.save_local(faiss_dir)
    print("✅ Local FAISS index updated successfully!")

if __name__ == "__main__":
    ingest_documents()