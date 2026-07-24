import os
from flask import Flask, render_template, redirect, session
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from logging_config import setup_logging, RequestIdMiddleware

# ============================================================
# --- CREATE APP ---
# ============================================================

app = Flask(__name__)
app.secret_key = "123456"

logger = setup_logging("api")
app.wsgi_app = RequestIdMiddleware(app.wsgi_app)

# ============================================================
# --- CONFIG ---
# ============================================================
app.config['WTF_CSRF_TIME_LIMIT'] = None
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 20,
    'max_overflow': 40,
    'pool_recycle': 1800,
    'pool_pre_ping': True,
    'pool_timeout': 30
}
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('SENDER_EMAIL')
app.config['MAIL_PASSWORD'] = os.getenv('GMAIL_APP_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('SENDER_EMAIL')
app.config['UPLOAD_FOLDER'] = "documents"
os.makedirs("documents", exist_ok=True)

# ============================================================
# --- INIT EXTENSIONS ---
# ============================================================

from extensions import db, mail, csrf
db.init_app(app)
mail.init_app(app)
csrf.init_app(app)

# ============================================================
# --- REGISTER BLUEPRINTS ---
# ============================================================

from routes.auth import auth_bp
from routes.chat import chat_bp
from routes.admin import admin_bp
from routes.escalation import escalation_bp

# ✅ Exempt auth blueprint from CSRF (user has no token yet at login)
csrf.exempt(auth_bp)

app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(escalation_bp)

# ============================================================
# --- PAGE ROUTES ---
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin-dashboard")
def admin_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/')
    from models.models import User
    user = User.query.get(user_id)
    if not user or not user.is_admin:
        return redirect('/')
    return render_template("dashboard.html")

@app.route("/dashboard-view")
def dashboard_view():
    if not session.get('user_id'):
        return redirect('/')
    return render_template("dashboard.html")

@app.route("/admin/documents-view")
def admin_documents_view():
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/')
    from models.models import User
    user = User.query.get(user_id)
    if not user or not user.is_admin:
        return redirect('/')
    return render_template("admin_documents.html")

@app.route("/admin/file-ids")
def get_file_ids():
    from models.models import ProcessedFile
    from flask import jsonify
    try:
        files = ProcessedFile.query.all()
        mapping = [{
            "file_id": f.file_id,
            "filename": f.filename,
            "file_type": f.filename.rsplit('.', 1)[1].upper() if '.' in f.filename else 'Unknown',
            "uploaded": str(f.processed_at)
        } for f in files]
        return jsonify({"success": True, "files": mapping})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ============================================================
# --- SCHEDULER ---
# ============================================================

def scheduled_cleanup():
    with app.app_context():
        from services.cache import cleanup_expired_cache
        cleanup_expired_cache()

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_cleanup, 'interval', hours=4)
scheduler.start()
print("✅ Cache cleanup scheduler started (runs every 4 hours)")

# ============================================================
# --- MAIN ---
# ============================================================

if __name__ == "__main__":
    with app.app_context():
        from extensions import db
        from models.models import User, ProcessedFile
        db.create_all()

        # Sync existing documents
        from routes.admin import save_processed_file_info

        UPLOAD_FOLDER = "documents"
        if os.path.exists(UPLOAD_FOLDER):
            admin_user = User.query.filter_by(is_admin=True).first()
            if admin_user:
                synced = 0
                newly_synced_filenames = []
                for filename in os.listdir(UPLOAD_FOLDER):
                    if '.' in filename and filename.rsplit('.', 1)[1].lower() == "pdf":
                        filepath = os.path.join(UPLOAD_FOLDER, filename)
                        if os.path.isfile(filepath) and not ProcessedFile.query.filter_by(filename=filename).first():
                            save_processed_file_info(admin_user.id, filename, filepath, chunk_count=0)
                            newly_synced_filenames.append(filename)
                            synced += 1

                if synced > 0:
                    print(f"✅ Synced {synced} documents to DB (registration only — "
                          f"ingestion is handled by ingest_worker.py, not startup)")

    from services.vectorstore import initialize_vectorstore
    initialize_vectorstore()
    app.run(debug=True, use_reloader=False)