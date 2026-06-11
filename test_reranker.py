from flashrank import Ranker, RerankRequest

reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="reranker_cache")

passages = [
    {"id": 0, "text": "The sky is blue because of light scattering."},
    {"id": 1, "text": "Python is a programming language."},
    {"id": 2, "text": "Neural networks are used in deep learning."},
]

request = RerankRequest(query="why is the sky blue", passages=passages)
results = reranker.rerank(request)

for r in results:
    print(f"Score: {r['score']} | {r['text'][:60]}")