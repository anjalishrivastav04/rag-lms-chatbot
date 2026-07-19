from dotenv import load_dotenv
load_dotenv()

import services.vectorstore as vs
vs.initialize_vectorstore()

query = "What exam city was allotted for the candidate?"

print("=== BM25 RESULTS ===")
for doc in vs.bm25_retriever.invoke(query):
    print(doc.metadata.get("source"), "-", doc.page_content[:60])

print("\n=== DENSE (ChromaDB) RESULTS ===")
for doc in vs.dense_retriever.invoke(query):
    print(doc.metadata.get("source"), "-", doc.page_content[:60])