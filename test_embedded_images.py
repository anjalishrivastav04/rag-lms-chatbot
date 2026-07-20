"""
test_embedded_images.py
------------------------
Standalone test for the new fitz-based embedded image extraction logic.
Does NOT touch FAISS, Redis, PostgreSQL, or run the full ingestion pipeline.

Usage:
    python test_embedded_images.py path/to/sample.pdf

What it does:
1. Opens the PDF once with PyMuPDF (fitz) - confirms fitz.open() works.
2. Loops through every page.
3. For each page, calls extract_embedded_image_text() and prints:
   - How many embedded images were found
   - How many passed the size filter (>5000 bytes)
   - The OCR text extracted from each (if any)
4. Closes the fitz doc in a finally block - confirms no leak even if a
   page throws an error.

Copy your real extract_text_from_image() import path below to match
your actual project structure before running.
"""

import sys
import os
import re
import fitz  # PyMuPDF

# 🔧 Adjust this import to match your actual project's ocr_handler location
try:
    from ocr_handler import extract_text_from_image
except ImportError:
    print("⚠️ Could not import extract_text_from_image from ocr_handler.")
    print("   Run this script from your rag-chatbot project root so imports resolve.")
    sys.exit(1)


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
                    print(f"      🔬 RAW confidence value: {confidence} (type: {type(confidence)})")
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_embedded_images.py path/to/sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Testing embedded image extraction on: {pdf_path}")
    print(f"{'='*70}\n")

    fitz_doc = None
    try:
        fitz_doc = fitz.open(pdf_path)
        print(f"✅ fitz.open() succeeded — {len(fitz_doc)} page(s) found\n")

        for page_num in range(len(fitz_doc)):
            print(f"--- Page {page_num + 1} ---")
            # In real ingest.py, main_text comes from PyPDFLoader's extracted
            # text for this page. Here we just use fitz's own text as a stand-in
            # for testing overlap-detection.
            page_text = fitz_doc[page_num].get_text().strip()
            blocks = extract_embedded_image_text(fitz_doc, page_num, page_text)

            if blocks:
                print(f"   📦 Total new OCR blocks for this page: {len(blocks)}")
                for i, b in enumerate(blocks):
                    preview = b[:150].replace("\n", " ")
                    print(f"      Block {i}: {preview}...")
            else:
                print(f"   (no new embedded-image text found)")
            print()

    except Exception as e:
        print(f"❌ Error opening/processing PDF: {e}")
    finally:
        if fitz_doc is not None:
            fitz_doc.close()
            print("✅ fitz_doc.close() called successfully — no leaked handle")


if __name__ == "__main__":
    main()