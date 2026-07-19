# 🛠️ Technology Stack — RAG Chatbot (C3i Hub, IIT Kanpur)

| Layer | Technology |
|---|---|
| **Backend Framework** | Flask (Python) |
| **Database** | PostgreSQL |
| **ORM** | SQLAlchemy / Flask-SQLAlchemy |
| **Caching** | Redis (exact-match cache, semantic cache backing, rate limiting, request locks, deleted-files blacklist, source index) |
| **Vector Store** | FAISS (document vectorstore + semantic cache store) |
| **Keyword Search** | BM25 (LangChain `BM25Retriever`) |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) |
| **Reranking** | FlashRank (`ms-marco-TinyBERT-L-2-v2`) |
| **Knowledge Graph** | Neo4j (Neo4j Desktop, local instance, Bolt protocol) |
| **LLM Provider** | Groq API |
| **LLM Models Used** | `llama-3.1-8b-instant`, `llama-3.1-8b-instant` |
| **Embeddings Model** | `all-MiniLM-L6-v2` (Sentence-Transformers + HuggingFace) |
| **OCR Engine** | EasyOCR |
| **Image/Vision Understanding** | Google Vision API |
| **Document Parsing** | LangChain document loaders/splitters, `langchain_core.documents.Document` |
| **Message Queue / Streaming** | Apache Kafka 4.3.0 (KRaft mode, no Zookeeper) |
| **Kafka Python Client** | `kafka-python` |
| **Java Runtime (for Kafka)** | Eclipse Temurin JDK 25 (OpenJDK) |
| **Background Job Scheduling** | APScheduler |
| **Authentication** | Flask sessions, Werkzeug password hashing (`generate_password_hash`/`check_password_hash`) |
| **File Handling** | Werkzeug `secure_filename`, Python `hashlib` (MD5 file hashing) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **CSS Framework** | Bootstrap 5 |
| **Markdown Rendering** | marked.js |
| **Voice Input** | Web Speech API (`SpeechRecognition`) |
| **Text-to-Speech** | Web Speech API (`SpeechSynthesisUtterance`) |
| **Translation** | Google Translate (unofficial endpoint, client-side fetch) |
| **Data Visualization (Dashboard)** | Chart.js (doughnut + bar charts) |
| **Version Control** | Git, GitHub |
| **Environment Management** | Python `venv`, `python-dotenv` |
| **Local Dev Server** | Flask development server (Werkzeug) |
