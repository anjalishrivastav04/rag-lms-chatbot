import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# --- APP CONFIG ---
# ============================================================
SECRET_KEY = "123456"

# --- FILE UPLOAD ---
UPLOAD_FOLDER = "documents"
ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS = {"pdf"}
OCR_SUPPORTED = set()

# --- POSTGRESQL ---
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_size': 20,
    'max_overflow': 40,
    'pool_recycle': 1800,
    'pool_pre_ping': True,
    'pool_timeout': 30
}

# --- REDIS ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_TTL = 3600

# --- MAIL ---
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USERNAME = os.getenv('SENDER_EMAIL')
MAIL_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
MAIL_DEFAULT_SENDER = os.getenv('SENDER_EMAIL')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

# --- RATE LIMITING ---
RATE_LIMIT_WINDOW = 4 * 60 * 60  # 4 hours in seconds
TOKEN_LIMIT_MAX = 50000

# --- CACHE ---
CACHE_DIR = "semantic_cache_store"
DISTANCE_THRESHOLD = 0.2

# --- TTL CONFIGURATION ---
CONTENT_TTL = {
    'lecture_notes': 24,
    'assignments': 7 * 24,
    'grades': 2,
    'syllabus': 90 * 24,
    'announcements': 12,
    'course_handbook': 60 * 24,
    'admin_feedback': 90 * 24,
}
DEFAULT_TTL = 24

# --- GROQ ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- VECTORSTORE ---
VECTORSTORE_DIR = "vectorstore"
RRF_K = 60