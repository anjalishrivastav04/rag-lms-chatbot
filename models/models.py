import uuid
from datetime import datetime, timedelta

# ============================================================
# --- DATABASE MODELS ---
# ============================================================

def get_db():
    from extensions import db
    return db

from extensions import db

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    cache_source = db.Column(db.String(50), default='NONE')
    response_time_ms = db.Column(db.Integer, default=0)
    ip_address = db.Column(db.String(45))  # IPv6-safe length
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "cache_source": self.cache_source,
            "response_time_ms": self.response_time_ms,
            "ip_address": self.ip_address,
            "created_at": str(self.created_at)
        }

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    ip_address = db.Column(db.String(45))  # last login IP
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    chats = db.relationship('ChatHistory', backref='user', lazy=True,
                            foreign_keys='ChatHistory.user_id',
                            cascade='all, delete-orphan')

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "ip_address": self.ip_address,
            "created_at": str(self.created_at)
        }

class RateLimit(db.Model):
    __tablename__ = 'rate_limits'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message_count = db.Column(db.Integer, default=0)
    window_start = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'message_count': self.message_count,
            'window_start': str(self.window_start)
        }

class ProcessedFile(db.Model):
    __tablename__ = 'processed_files'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.String(36), unique=True, nullable=False,
                        default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    file_size = db.Column(db.Integer)
    processed_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    chunk_count = db.Column(db.Integer, default=0)
    version = db.Column(db.String(50))

    def to_dict(self):
        return {
            "filename": self.filename,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "processed_at": str(self.processed_at),
            "chunk_count": self.chunk_count,
            "version": self.version
        }

class ChunkingDecision(db.Model):          # ← add here, right after ProcessedFile
    __tablename__ = 'chunking_decisions'
    content_hash = db.Column(db.String(64), primary_key=True)
    method = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'content_hash': self.content_hash,
            'method': self.method,
            'confidence': self.confidence,
            'created_at': str(self.created_at)
        }

class SemanticCacheRecord(db.Model):
    __tablename__ = 'semantic_cache'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    query_text = db.Column(db.Text, nullable=False)
    query_embedding = db.Column(db.Text)
    response = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(50), default='general')
    hit_count = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)

    def is_expired(self):
        return datetime.utcnow() > self.expires_at

    def time_remaining(self):
        if self.is_expired():
            return "Expired"
        delta = self.expires_at - datetime.utcnow()
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f} hours"

    def to_dict(self):
        return {
            'id': self.id,
            'query': self.query_text,
            'response': self.response,
            'content_type': self.content_type,
            'hits': self.hit_count,
            'created': str(self.created_at),
            'expires': str(self.expires_at),
            'ttl_remaining': self.time_remaining()
        }

class ResponseEvaluation(db.Model):
    __tablename__ = 'response_evaluations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(255), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text)
    score = db.Column(db.Integer)
    feedback = db.Column(db.Text)
    evaluated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'score': self.score,
            'feedback': self.feedback,
            'evaluated_at': str(self.evaluated_at)
        }

class AdminFeedback(db.Model):
    __tablename__ = 'admin_feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    original_answer = db.Column(db.Text, nullable=False)
    feedback_type = db.Column(db.String(10), nullable=False)
    correct_answer = db.Column(db.Text)
    is_ingested = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'original_answer': self.original_answer,
            'feedback_type': self.feedback_type,
            'correct_answer': self.correct_answer,
            'is_ingested': self.is_ingested,
            'created_at': str(self.created_at)
        }

class EscalationRequest(db.Model):
    __tablename__ = 'escalation_requests'
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, nullable=False)
    user_email = db.Column(db.String(120), nullable=False)
    question = db.Column(db.Text, nullable=False)
    ai_answer = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.Text)
    is_resolved = db.Column(db.Boolean, default=False)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    resolved_at = db.Column(db.DateTime(timezone=True))

    def to_dict(self):
        return {
            'id': self.id,
            'user_email': self.user_email,
            'question': self.question,
            'ai_answer': self.ai_answer,
            'correct_answer': self.correct_answer,
            'is_resolved': self.is_resolved,
            'ip_address': self.ip_address,
            'created_at': str(self.created_at)
        }