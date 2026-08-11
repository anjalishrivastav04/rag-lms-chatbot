import os
from dotenv import load_dotenv

load_dotenv()


def get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if provider in {"gemini", "google", "g"} and gemini_key:
        return "gemini"
    if provider in {"groq", "grok"} and groq_key:
        return "groq"
    if gemini_key:
        return "gemini"
    if groq_key:
        return "groq"
    return "groq"


def create_chat_model(temperature: float = 0.0, model: str | None = None):
    provider = get_llm_provider()

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the 'langchain-google-genai' package."
            ) from exc

        selected_model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
        )

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError(
            "Groq support requires the 'langchain-groq' package."
        ) from exc

    selected_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=selected_model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY", ""),
        timeout=30,       # ✅ Don't hang forever — fail fast after 30s
        max_retries=2,    # ✅ Retry twice before giving up
    )
