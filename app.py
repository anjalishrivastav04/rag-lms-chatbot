import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

app = Flask(__name__)

# --- CORE EMBEDDINGS & MAIN DOCUMENT RETRIEVER ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- CUSTOM LOCAL SEMANTIC CACHE CONFIG ---
CACHE_DIR = "semantic_cache_store"
DISTANCE_THRESHOLD = 0.2  # Closer to 0 means stricter. 0.2 allows for slight variations in phrasing.

def check_semantic_cache(question):
    """Checks the local cache vectorstore for a semantically similar question."""
    if not os.path.exists(CACHE_DIR):
        return None
    try:
        cache_store = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
        # Search for the single most similar past question
        results_with_scores = cache_store.similarity_search_with_score(question, k=1)
        
        if results_with_scores:
            cached_doc, distance = results_with_scores[0]
            # FAISS L2 Distance: lower means closer. 
            if distance <= DISTANCE_THRESHOLD:
                print(f"🎯 [CACHE HIT] Found similar query with distance {distance:.3f}")
                return cached_doc.metadata.get("response")
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")
    return None

def save_to_semantic_cache(question, response_text):
    """Saves a new question and its corresponding LLM response to the cache index."""
    new_cache_doc = Document(
        page_content=question,
        metadata={"response": response_text}
    )
    try:
        if os.path.exists(CACHE_DIR):
            cache_store = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
            cache_store.add_documents([new_cache_doc])
            cache_store.save_local(CACHE_DIR)
        else:
            cache_store = FAISS.from_documents([new_cache_doc], embeddings)
            cache_store.save_local(CACHE_DIR)
        print("💾 [CACHE UPDATE] Saved new response to semantic cache.")
    except Exception as e:
        print(f"⚠️ Cache write error: {e}")

# --- GROQ LLM CONFIG ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question):
    # 1. Check if an answer already exists semantically
    cached_response = check_semantic_cache(question)
    if cached_response:
        return cached_response

    # 2. Cache Miss: Run your standard RAG retrieval pipeline
    print("⚡ [CACHE MISS] Fetching fresh answer from Groq...")
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful assistant. Answer the question based on the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    answer = response.content

    # 3. Save the newly generated answer to the cache for future hits
    save_to_semantic_cache(question, answer)
    return answer

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    try:
        reply = get_answer(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)