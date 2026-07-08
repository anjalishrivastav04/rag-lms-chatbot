from app import app

with app.app_context():
    from services.vectorstore import initialize_vectorstore
    initialize_vectorstore()

    import services.vectorstore as vs_module
    from collections import defaultdict

    by_filetype = defaultdict(set)
    for doc in vs_module.ALL_DOCS:
        filetype = doc.metadata.get("filetype", "unknown")
        source = doc.metadata.get("source", "unknown")
        by_filetype[filetype].add(source)

    print("--- Distinct filetype values and their sources ---")
    for filetype, sources in by_filetype.items():
        print(f"\nfiletype = '{filetype}' ({len(sources)} sources):")
        for s in sorted(sources):
            print("  -", s)