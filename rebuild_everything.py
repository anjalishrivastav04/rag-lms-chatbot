# rebuild_everything.py
import os
from ingest import ingest_documents

DOCUMENTS_FOLDER = "documents"

files = [f for f in os.listdir(DOCUMENTS_FOLDER) if os.path.isfile(os.path.join(DOCUMENTS_FOLDER, f))]

print(f"Found {len(files)} files to rebuild.")

for i, filename in enumerate(files, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(files)}] Rebuilding: {filename}")
    print(f"{'='*60}")
    try:
        ingest_documents(filename)
    except Exception as e:
        print(f"⚠️ Failed to rebuild {filename}: {e}")

print("\nDone.")