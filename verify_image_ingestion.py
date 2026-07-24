"""
Standalone verification script — run AFTER ingest.py to confirm
image captioning/OCR metadata landed correctly in ChromaDB.

Usage:
    python verify_image_ingestion.py
    python verify_image_ingestion.py <filename.pdf>   # filter to one file
"""
import sys
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "rag_documents"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000


def main():
    filter_filename = sys.argv[1] if len(sys.argv) > 1 else None

    print("🔌 Connecting to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    where_clause = {"source": filter_filename} if filter_filename else None
    raw = vectorstore.get(
        where=where_clause,
        include=["documents", "metadatas"]
    )

    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    if not docs:
        print("⚠️ No chunks found. Did ingestion run successfully?")
        return

    print(f"📊 Total chunks found: {len(docs)}\n")

    caption_chunks = []
    embedded_text_chunks = []
    plain_chunks = []

    for content, meta in zip(docs, metas):
        meta = meta or {}
        if meta.get("has_image_caption"):
            caption_chunks.append((content, meta))
        elif meta.get("has_embedded_image_text"):
            embedded_text_chunks.append((content, meta))
        else:
            plain_chunks.append((content, meta))

    print(f"🖼️  Chunks with VLM image captions : {len(caption_chunks)}")
    print(f"📝  Chunks with embedded OCR only   : {len(embedded_text_chunks)}")
    print(f"📄  Plain text chunks (no images)   : {len(plain_chunks)}")
    print()

    if caption_chunks:
        print("=" * 80)
        print("SAMPLE: chunks with VLM captions")
        print("=" * 80)
        for content, meta in caption_chunks[:3]:
            print(f"\n📄 Source: {meta.get('source')} | Page: {meta.get('page', 'N/A')} | Chunk: {meta.get('chunk_index')}")
            if "[IMAGE DESCRIPTION]" in content:
                caption_part = content.split("[IMAGE DESCRIPTION]:", 1)[1].strip()
                print(f"   Caption text: {caption_part[:300]}{'...' if len(caption_part) > 300 else ''}")
            else:
                print("   ⚠️ has_image_caption=True but [IMAGE DESCRIPTION] marker not found in content — check wiring.")
            print("-" * 80)

    if embedded_text_chunks:
        print("\n" + "=" * 80)
        print("SAMPLE: chunks with embedded OCR text only (no VLM call)")
        print("=" * 80)
        for content, meta in embedded_text_chunks[:2]:
            print(f"\n📄 Source: {meta.get('source')} | Page: {meta.get('page', 'N/A')} | Chunk: {meta.get('chunk_index')}")
            snippet = content[-300:] if len(content) > 300 else content
            print(f"   Tail content: ...{snippet}")
            print("-" * 80)

    print("\n✅ Verification complete.")
    if not caption_chunks and not embedded_text_chunks:
        print("⚠️ No image-derived chunks found at all. Either the PDF had no embedded images,")
        print("   or ingestion ran before the vision_handler wiring was added.")


if __name__ == "__main__":
    main()
    