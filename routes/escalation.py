import secrets
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from extensions import db, mail, csrf
from models.models import EscalationRequest
from services.rag import ingest_feedback_to_vectorstore
from services.cache import add_to_blacklist
from extensions import redis_client
from config import ADMIN_EMAIL, REDIS_TTL
from flask_mail import Message as MailMessage

escalation_bp = Blueprint('escalation', __name__)

# ✅ Reuse the "api" logger — app.py already ran setup_logging("api"), and
# RequestIdMiddleware assigns a trace_id to every incoming request, so these
# log lines are automatically traceable back to the originating chat query.
logger = logging.getLogger("api")

# ============================================================
# --- ESCALATION ROUTES ---
# ============================================================

@csrf.exempt
@escalation_bp.route("/escalate", methods=["POST"])
def escalate():
    data = request.get_json() or {}
    user_email = data.get("email", "").strip()
    question = data.get("question", "").strip()
    ai_answer = data.get("ai_answer", "").strip()

    if not user_email or not question:
        return jsonify({"success": False, "message": "Missing email or question!"})

    try:
        token = secrets.token_urlsafe(32)
        escalation = EscalationRequest(
            token=token,
            user_email=user_email,
            question=question,
            ai_answer=ai_answer,
            ip_address=request.remote_addr
        )
        db.session.add(escalation)
        db.session.commit()

        logger.warning("escalation_created", extra={
            "escalation_id": escalation.id,
            "user_email": user_email,
            "question": question[:100],
            "ip_address": request.remote_addr,
        })

        resolve_link = f"http://localhost:5000/admin/resolve/{token}"

        msg = MailMessage(
            subject="🚨 RAGBot Escalation — Low Confidence Answer",
            recipients=[ADMIN_EMAIL]
        )
        msg.html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #1e40af;">🤖 RAGBot — Admin Help Required</h2>
            <p>A user needs help with a question the bot couldn't answer confidently.</p>
            <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; margin: 16px 0;">
                <strong>👤 User Email:</strong> {user_email}
            </div>
            <div style="background: #fef3c7; padding: 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #f59e0b;">
                <strong>❓ Question:</strong><br/>
                <p style="margin-top: 8px;">{question}</p>
            </div>
            <div style="background: #fee2e2; padding: 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #ef4444;">
                <strong>🤖 AI's Answer (Low Confidence):</strong><br/>
                <p style="margin-top: 8px;">{ai_answer}</p>
            </div>
            <a href="{resolve_link}"
               style="display: inline-block; background: #1e40af; color: white; padding: 12px 24px;
                      border-radius: 8px; text-decoration: none; font-weight: bold; margin-top: 16px;">
                ✅ Click Here to Provide Correct Answer
            </a>
            <p style="color: #64748b; font-size: 12px; margin-top: 24px;">
                This link is unique and will only work once.
            </p>
        </div>
        """
        mail.send(msg)
        logger.info("escalation_email_sent", extra={
            "escalation_id": escalation.id,
            "recipient": ADMIN_EMAIL,
        })
        return jsonify({"success": True, "message": "✅ Sent to admin successfully!"})

    except Exception as e:
        db.session.rollback()
        logger.error("escalation_creation_failed", exc_info=True, extra={
            "user_email": user_email,
            "question": question[:100],
        })
        return jsonify({"success": False, "message": str(e)})


@escalation_bp.route("/admin/resolve/<token>", methods=["GET", "POST"])
def admin_resolve(token):
    escalation = EscalationRequest.query.filter_by(token=token).first()

    if not escalation:
        logger.warning("resolve_link_invalid", extra={"token": token[:12] + "..."})
        return "<h2>❌ Invalid or expired link.</h2>", 404

    if escalation.is_resolved:
        logger.info("resolve_link_already_resolved", extra={"escalation_id": escalation.id})
        return "<h2>✅ This question has already been resolved.</h2>", 200

    if request.method == "GET":
        logger.info("resolve_page_viewed", extra={"escalation_id": escalation.id})
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAGBot — Resolve Question</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; background: #f8fafc; }}
                .card {{ background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .question {{ background: #fef3c7; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b; }}
                .ai-answer {{ background: #fee2e2; padding: 16px; border-radius: 8px; border-left: 4px solid #ef4444; }}
                textarea {{ width: 100%; min-height: 150px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px; font-family: Arial; resize: vertical; box-sizing: border-box; }}
                button {{ background: #1e40af; color: white; padding: 12px 28px; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; margin-top: 12px; }}
                button:hover {{ background: #1e3a8a; }}
                label {{ font-weight: bold; color: #334155; display: block; margin-bottom: 8px; }}
                h1 {{ color: #1e40af; }}
            </style>
        </head>
        <body>
            <h1>🤖 RAGBot — Admin Resolution Panel</h1>
            <div class="card">
                <label>👤 User Email:</label>
                <p>{escalation.user_email}</p>
            </div>
            <div class="card question">
                <label>❓ Question:</label>
                <p>{escalation.question}</p>
            </div>
            <div class="card ai-answer">
                <label>🤖 AI's Low-Confidence Answer:</label>
                <p>{escalation.ai_answer}</p>
            </div>
            <div class="card">
                <label>✅ Your Correct Answer:</label>
                <textarea id="correctAnswer" placeholder="Type the correct answer here..."></textarea>
                <button onclick="submitAnswer()">Submit & Add to Knowledge Base</button>
                <p id="status" style="margin-top: 12px; color: green;"></p>
            </div>
            <script>
                async function submitAnswer() {{
                    const answer = document.getElementById('correctAnswer').value.trim();
                    if (!answer) {{ alert('Please provide an answer!'); return; }}
                    const res = await fetch('/admin/resolve/{token}', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{ correct_answer: answer }})
                    }});
                    const data = await res.json();
                    if (data.success) {{
                        document.getElementById('status').textContent = '✅ Answer submitted and added to knowledge base!';
                        document.querySelector('button').disabled = true;
                    }} else {{
                        document.getElementById('status').textContent = '❌ ' + data.message;
                        document.getElementById('status').style.color = 'red';
                    }}
                }}
            </script>
        </body>
        </html>
        """

    if request.method == "POST":
        data = request.get_json() or {}
        correct_answer = data.get("correct_answer", "").strip()
        if not correct_answer:
            return jsonify({"success": False, "message": "Please provide an answer!"})

        try:
            escalation.correct_answer = correct_answer
            escalation.is_resolved = True
            escalation.resolved_at = datetime.utcnow()
            db.session.commit()

            logger.info("escalation_resolved", extra={
                "escalation_id": escalation.id,
                "question": escalation.question[:100],
            })

            feedback_id = f"escalation_{escalation.id}"
            ingest_success = ingest_feedback_to_vectorstore(
                escalation.question, correct_answer, feedback_id
            )
            logger.info("escalation_feedback_ingested", extra={
                "escalation_id": escalation.id,
                "ingest_success": ingest_success,
            })

            cache_key = f"rag:{escalation.question.lower().strip()}"
            redis_client.setex(cache_key, REDIS_TTL, json.dumps(correct_answer))

            try:
                user_msg = MailMessage(
                    subject="✅ Your RAGBot Question Has Been Answered!",
                    recipients=[escalation.user_email]
                )
                user_msg.html = f"""
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #1e40af;">✅ Your Question Has Been Answered!</h2>
                    <div style="background: #fef3c7; padding: 16px; border-radius: 8px; margin: 16px 0;">
                        <strong>❓ Your Question:</strong><br/>
                        <p style="margin-top: 8px;">{escalation.question}</p>
                    </div>
                    <div style="background: #dcfce7; padding: 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #22c55e;">
                        <strong>✅ Admin's Answer:</strong><br/>
                        <p style="margin-top: 8px;">{correct_answer}</p>
                    </div>
                    <p style="color: #64748b; font-size: 13px;">
                        This answer has also been added to the RAGBot knowledge base!
                    </p>
                </div>
                """
                mail.send(user_msg)
                logger.info("resolution_email_sent", extra={
                    "escalation_id": escalation.id,
                    "recipient": escalation.user_email,
                })
            except Exception:
                logger.warning("resolution_email_failed", exc_info=True, extra={
                    "escalation_id": escalation.id,
                    "recipient": escalation.user_email,
                })

            return jsonify({
                "success": True,
                "message": "✅ Answer saved and added to knowledge base!",
                "ingested": ingest_success
            })

        except Exception as e:
            db.session.rollback()
            logger.error("escalation_resolve_failed", exc_info=True, extra={
                "escalation_id": escalation.id,
            })
            return jsonify({"success": False, "message": str(e)})