from extensions import db, eval_llm
from models.models import ResponseEvaluation
from services.rag import safe_invoke

# ============================================================
# --- RAG RESPONSE EVALUATOR ---
# ============================================================

def evaluate_response(question, answer, context, user_id, session_id):
    try:
        eval_prompt = f"""You are a RAG system evaluator. Evaluate the answer fairly and practically.

QUESTION: {question}

RETRIEVED CONTEXT: {context[:1000]}

GENERATED ANSWER: {answer}

Evaluation criteria:
- RELEVANCE: Does the answer address the question asked?
- FAITHFULNESS: Is the answer based on the context (summarizing is fine, no need to copy word for word)?
- COMPLETENESS: Does the answer provide enough useful information?
- CLARITY: Is the answer clean and easy to understand?

IMPORTANT RULES:
- If the answer summarizes context in its own words, that is GOOD not bad
- If the answer combines info from multiple chunks, that is GOOD
- Only penalize if the answer makes up facts NOT in the context
- Only penalize if the answer completely ignores the question

Rate strictly on scale 1-5:
1 = Completely wrong or made up
2 = Partially relevant but missing key info
3 = Relevant and mostly correct
4 = Good answer, faithful to context
5 = Perfect answer, complete and accurate

Reply in this EXACT format only:
SCORE: <number 1-5>
FEEDBACK: <one sentence explanation>

Nothing else."""

        eval_response = safe_invoke(eval_llm, eval_prompt)
        content = eval_response.content.strip()
        lines = content.split('\n')
        score = 3
        feedback = "Evaluation completed"
        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = int(line.replace('SCORE:', '').strip())
                    score = max(1, min(5, score))
                except:
                    score = 3
            elif line.startswith('FEEDBACK:'):
                feedback = line.replace('FEEDBACK:', '').strip()

        evaluation = ResponseEvaluation(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer,
            context=context[:2000],
            score=score,
            feedback=feedback
        )
        db.session.add(evaluation)
        db.session.commit()
        print(f"✅ Response evaluated — Score: {score}/5 | {feedback}")
        return score, feedback

    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        db.session.rollback()
        return None, None