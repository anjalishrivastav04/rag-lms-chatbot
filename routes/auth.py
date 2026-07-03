import logging
from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
from models.models import User
from extensions import db, csrf

auth_bp = Blueprint('auth', __name__)

# ✅ Same "api" logger as everywhere else in the Flask process — RequestIdMiddleware
# already tags each request with a trace_id, so these lines are traceable too.
logger = logging.getLogger("api")


@csrf.exempt
@auth_bp.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not username or not email or not password:
        return jsonify({"success": False, "message": "All fields required!"})
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be 6+ characters!"})
    if User.query.filter_by(username=username).first():
        logger.warning("signup_failed_duplicate_username", extra={"username": username})
        return jsonify({"success": False, "message": "Username already exists!"})
    if User.query.filter_by(email=email).first():
        logger.warning("signup_failed_duplicate_email", extra={"email": email})
        return jsonify({"success": False, "message": "Email already registered!"})

    try:
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        session['is_admin'] = new_user.is_admin

        logger.info("signup_success", extra={
            "user_id": new_user.id,
            "username": username,
            "ip_address": request.remote_addr,
        })
        return jsonify({"success": True, "message": f"Welcome {username}! 🎉", "user": new_user.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error("signup_error", exc_info=True, extra={"username": username})
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@csrf.exempt
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required!"})

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        # ✅ Log failed attempts — this is your signal for brute-force /
        # credential-stuffing attempts. Never logs the password itself.
        logger.warning("login_failed", extra={
            "username": username,
            "ip_address": request.remote_addr,
            "reason": "user_not_found" if not user else "bad_password",
        })
        return jsonify({"success": False, "message": "Invalid username or password!"})

    user.ip_address = request.remote_addr
    db.session.commit()
    session['user_id'] = user.id
    session['username'] = user.username
    session['is_admin'] = user.is_admin

    logger.info("login_success", extra={
        "user_id": user.id,
        "username": username,
        "is_admin": user.is_admin,
        "ip_address": request.remote_addr,
    })
    return jsonify({"success": True, "message": f"Welcome back, {username}! 👋", "user": user.to_dict()})


@csrf.exempt
@auth_bp.route("/logout", methods=["POST"])
def logout():
    user_id = session.get('user_id')
    username = session.get('username')
    session.clear()
    logger.info("logout", extra={"user_id": user_id, "username": username})
    return jsonify({"success": True, "message": "Logged out successfully!"})


@csrf.exempt
@auth_bp.route("/current-user", methods=["GET"])
def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Not logged in"})
    user = User.query.get(user_id)
    if not user:
        logger.warning("session_user_not_found", extra={"user_id": user_id})
        session.clear()
        return jsonify({"success": False, "message": "User not found"})
    return jsonify({"success": True, "user": user.to_dict()})