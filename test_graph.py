from app import app
from graph_answer import build_graph

with app.app_context():
    from services.vectorstore import initialize_vectorstore
    initialize_vectorstore()
    print("📚 Vectorstore initialized for test.")

    graph = build_graph()
    result = graph.invoke({
        "question": "what topics are covered in the python syllabus",
        "session_id": "test-session",
        "user_id": 1
    })
    print("ANSWER:", result["answer"])
    print("CONFIDENCE:", result.get("confidence"))
    print("ESCALATION NEEDED:", result.get("needs_escalation"))