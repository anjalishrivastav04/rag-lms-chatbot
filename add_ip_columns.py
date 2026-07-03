"""
One-time script to add ip_address columns to existing tables.
Run this once: python add_ip_columns.py
Safe to re-run — uses IF NOT EXISTS.

Does NOT import app.py on purpose — importing app.py triggers DB queries
against models that already expect these columns, before the columns exist.
"""
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

from extensions import db
import config

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    print("✨ Adding ip_address columns...")
    db.session.execute(db.text("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);
    """))
    db.session.execute(db.text("""
        ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);
    """))
    db.session.execute(db.text("""
        ALTER TABLE escalation_requests ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);
    """))
    db.session.commit()
    print("✅ ip_address columns added to users, chat_history, escalation_requests")