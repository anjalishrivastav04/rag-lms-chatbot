"""
ingest/extractors.py
--------------------
Document extraction layer: handles .txt, .pdf (with embedded-image OCR
fallback), and image files (.jpg/.png/…). Returns a flat list of
LangChain Documents ready for chunking.
"""

import os
import re

import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from ocr_handler import save_ocr_text_to_file, extract_text_from_image
from vision_handler import analyze_image

# ── Poppler (Windows / OneDrive path — no-op on Linux) ───────────────────────
POPPLER_PATH = r"C:\Users\iaman\OneDrive\Documents\Desktop\poppler-26.02.0\Library\bin"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


class DocumentExtractor:
    """
    Loads a file from disk and returns a list of LangChain Documents.

    Supports:
    - .txt  — direct text load
    - .pdf  — PyPDF text extraction + embedded-image OCR + OCR fallback for
              scanned/broken pages
    - images — optional VLM analysis + Tesseract OCR
    """

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, path: str) -> list[Document]:
        """Dispatch to the correct loader based on file extension."""
        filename_lower = os.path.basename(path).lower()

        if filename_lower.endswith('.txt'):
            return self._load_txt(path)
        elif filename_lower.endswith('.pdf'):
            return self._load_pdf(path)
        elif any(filename_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            return self._load_image(path)
        else:
            print(f"⚠️ Unsupported file type: {path}")
            return []

    # ── Private loaders ─────────────────────────────────────────────────────

    def _load_txt(self, path: str) -> list[Document]:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    def _load_pdf(self, path: str) -> list[Document]:
        """
        Per-page hybrid extraction:
        - Good pages  → PyPDF text + embedded-image OCR merged in
        - Broken pages → Poppler render → Tesseract OCR fallback
        """
        filename = os.path.basename(path)
        print("🔍 Loading PDF layout layers...")
        loader = PyPDFLoader(path)
        raw_pdf_docs = loader.load()
        docs: list[Document] = []

        for page_num, doc in enumerate(raw_pdf_docs):
            text_content = doc.page_content.strip()
            clean_char_count = len(re.sub(r'\s+', '', text_content))

            # Detect "spaced-out" text (OCR artefact where every char is spaced)
            is_spaced_out = False
            if len(text_content) > 0:
                single_letters = len(re.findall(r'\b[a-zA-Z]\b', text_content))
                total_words = max(len(text_content.split()), 1)
                if (single_letters / total_words) > 0.40:
                    is_spaced_out = True

            if clean_char_count > 30 and not is_spaced_out:
                docs.append(doc)
                # Also extract text from any embedded images on this page
                embedded_blocks = self._extract_embedded_images(path, page_num, text_content)
                if embedded_blocks:
                    print(f"🖼️ Page {page_num + 1}: {len(embedded_blocks)} embedded image(s) with extractable text — merging.")
                    docs.append(Document(
                        page_content=(
                            f"[Content extracted from an embedded image/diagram on page {page_num + 1} of {filename}]\n"
                            + "\n\n".join(embedded_blocks)
                        ),
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "has_embedded_image_text": True,
                            "is_embedded_image_chunk": True,
                        },
                    ))
            else:
                docs.extend(self._ocr_fallback(path, filename, page_num, doc))

        return docs

    def _ocr_fallback(self, path: str, filename: str, page_num: int, original_doc: Document) -> list[Document]:
        """Render a single PDF page to an image and run Tesseract OCR on it."""
        print(f"📸 Page {page_num + 1} flagged as image scan or broken layout. Initiating OCR fallback...")
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,
                poppler_path=POPPLER_PATH,
            )
            if not images:
                print(f"⚠️ No image rendered for Page {page_num + 1}. Falling back to original text.")
                original_doc.metadata["degraded"] = True
                original_doc.metadata["degraded_reason"] = "no_image_rendered"
                return [original_doc]

            temp_img_path = f"documents/temp_page_{page_num + 1}.jpg"
            images[0].save(temp_img_path, 'JPEG')
            ocr_text, _confidence = extract_text_from_image(temp_img_path)
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            if ocr_text and ocr_text.strip():
                print(f"✨ Successfully extracted clean text from PDF Page {page_num + 1}!")
                return [Document(
                    page_content=ocr_text,
                    metadata={"source": filename, "page": page_num + 1},
                )]
            else:
                print(f"⚠️ OCR returned empty text for Page {page_num + 1}. Using original (degraded) text.")
                original_doc.metadata["degraded"] = True
                original_doc.metadata["degraded_reason"] = "ocr_empty"
                return [original_doc]

        except Exception as err:
            print(f"⚠️ Image-to-PDF parser skipped Page {page_num + 1}: {err}")
            original_doc.metadata["degraded"] = True
            original_doc.metadata["degraded_reason"] = f"ocr_exception: {err}"
            return [original_doc]

    def _load_image(self, path: str) -> list[Document]:
        """Run optional VLM analysis + Tesseract OCR on a standalone image file."""
        filename = os.path.basename(path)
        vision_description = "Visual descriptions disabled via system config parameters."

        if os.getenv("USE_GOOGLE_VISION", "False").lower() == "true":
            print("🖼️ Image detected — attempting analysis with local VLM engine...")
            try:
                vision_description = analyze_image(path)
            except Exception:
                print("⚠️ Local VLM parsing failed. Bypassing structural description...")
                vision_description = "Visual workspace descriptions unavailable. Falling back to clean OCR."
        else:
            print("⚙️ Skipping VLM engine (USE_GOOGLE_VISION=False). Running local OCR framework pipeline...")

        print("📸 Running OCR for handwritten text...")
        output_txt_path, _ocr_error = save_ocr_text_to_file(path)

        combined_text = f"=== IMAGE ANALYSIS SUMMARY ===\n{vision_description}\n\n"

        if output_txt_path and os.path.exists(output_txt_path):
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            combined_text += f"=== OCR EXTRACTED TEXT ===\n{ocr_text}"
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            loader = TextLoader(output_txt_path, encoding="utf-8")
        else:
            temp_vision_path = f"documents/{os.path.splitext(filename)[0]}_vision.txt"
            with open(temp_vision_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            loader = TextLoader(temp_vision_path, encoding="utf-8")

        return loader.load()

    def _extract_embedded_images(self, pdf_path: str, page_num: int, main_text: str) -> list[str]:
        """
        Find embedded raster images on a PDF page and run OCR on each
        cluster (handles PDFs that slice one image into many thin strips).
        Returns a list of non-duplicate OCR text blocks.
        """
        ocr_blocks: list[str] = []
        try:
            pdf_doc = fitz.open(pdf_path)
            page = pdf_doc[page_num]
            image_list = page.get_images(full=True)

            if not image_list:
                pdf_doc.close()
                return ocr_blocks

            # Collect on-page bounding boxes
            all_rects = []
            for img_info in image_list:
                xref = img_info[0]
                for r in page.get_image_rects(xref):
                    all_rects.append(r)

            if not all_rects:
                pdf_doc.close()
                return ocr_blocks

            # Cluster vertically adjacent strips into unified regions
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
                if region_rect.width < 20 or region_rect.height < 20:
                    continue  # skip tiny decorative elements
                try:
                    mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
                    pix = page.get_pixmap(matrix=mat, clip=region_rect)
                    temp_img_path = f"documents/temp_embedded_{page_num + 1}_{region_index}.png"
                    pix.save(temp_img_path)

                    ocr_text, _confidence = extract_text_from_image(temp_img_path)
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

                    if not ocr_text or not ocr_text.strip():
                        continue

                    # Skip if mostly duplicate of page text
                    ocr_words = set(re.findall(r'\w+', ocr_text.lower()))
                    main_words = set(re.findall(r'\w+', main_text.lower()))
                    if ocr_words and len(ocr_words & main_words) / len(ocr_words) > 0.7:
                        continue

                    ocr_blocks.append(ocr_text.strip())
                except Exception as region_err:
                    print(f"⚠️ Could not process embedded image region {region_index} on page {page_num + 1}: {region_err}")

            pdf_doc.close()
        except Exception as e:
            print(f"⚠️ Could not open PDF for embedded image extraction: {e}")

        return ocr_blocks
