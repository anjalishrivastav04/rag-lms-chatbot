from app import app
from extensions import db
from sqlalchemy import text

with app.app_context():
    try:
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        db.session.commit()
        print("✅ pgvector extension is available and enabled.")
    except Exception as e:
        db.session.rollback()
        print("❌ pgvector NOT available on this Postgres instance.")
        print("Error:", e)