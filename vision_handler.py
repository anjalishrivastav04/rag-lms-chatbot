import os
import requests
import base64

def analyze_image(image_path):
    """
    Generates a structural and visual description of an image 
    using a locally hosted Ollama + LLaVA vision model pipeline.
    """
    if not os.path.exists(image_path):
        return f"Error: Image path '{image_path}' does not exist."

    try:
        print(f"🧠 Encoding image for local VLM inference...")
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Point to your local Ollama server running LLaVA
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava",
            "prompt": (
                "Describe this image in detail for a RAG database index. "
                "Explicitly mention primary subjects, colors, layout structure, text regions, "
                "and any distinct technical data visible."
            ),
            "images": [encoded_string],
            "stream": False
        }
        
        print(f"🤖 Sending request to local LLaVA engine (this may take a few seconds)...")
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            description = response.json().get("response", "").strip()
            print(f"✨ Local vision analysis complete!")
            return description
        else:
            print(f"⚠️ Ollama returned status code: {response.status_code}")
            return f"Local Vision Error: Server returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Refused: Is Ollama running?")
        return "Visual analysis bypassed: Local Ollama service is not running."
    except Exception as e:
        print(f"❌ Unexpected vision processing exception: {str(e)}")
        return f"Failed local vision analysis: {str(e)}"

# -------------------------------------------
# CONFIG-DRIVEN CAPTION WRAPPER (backend swappable via env var)
# -------------------------------------------
VLM_BACKEND = os.getenv("VLM_BACKEND", "mock")  # "mock" | "ollama" | "vllm"
VLM_ENDPOINT = os.getenv("VLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")  # used when VLM_BACKEND=vllm
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")  # used when VLM_BACKEND=vllm

CAPTION_PROMPT = (
    "Describe this image in detail for a RAG database index. "
    "Explicitly mention primary subjects, colors, layout structure, text regions, "
    "and any distinct technical data visible."
)

def _mock_caption(image_path):
    """Stand-in used while waiting for GPU access — lets the whole pipeline
    be built and tested end-to-end before the real VLM is connected."""
    print(f"🧪 [MOCK MODE] Skipping real VLM call for {image_path}")
    return "Mock caption: diagram/image content placeholder (VLM backend not connected yet)."

def _vllm_caption(image_path):
    """Calls a self-hosted vLLM server (OpenAI-compatible), e.g. Qwen3-VL on the GPU machine."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded_string}"}}
                    ]
                }
            ],
            "max_tokens": 512
        }
        response = requests.post(VLM_ENDPOINT, json=payload, timeout=300)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"⚠️ vLLM endpoint returned status code: {response.status_code}")
            return f"Vision Error: vLLM server returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        print("❌ Connection Refused: Is the vLLM server running/reachable?")
        return "Visual analysis bypassed: vLLM service is not reachable."
    except Exception as e:
        print(f"❌ Unexpected vLLM captioning exception: {str(e)}")
        return f"Failed vLLM vision analysis: {str(e)}"

def get_image_caption(image_path):
    """
    Single entry point ingest.py should call. Backend is controlled
    entirely by the VLM_BACKEND env var:
      - "mock"   -> dummy text, zero dependencies, use this NOW
      - "ollama" -> uses your existing analyze_image() / LLaVA, unchanged
      - "vllm"   -> real GPU-hosted model once access comes through
    Switching later = change one env var, no code edits.
    """
    if VLM_BACKEND == "vllm":
        return _vllm_caption(image_path)
    elif VLM_BACKEND == "ollama":
        return analyze_image(image_path)
    else:
        return _mock_caption(image_path)


# -------------------------------------------
# IMAGE TYPE CLASSIFIER (text-dense vs diagram routing)
# -------------------------------------------
def classify_image_type(ocr_text, ocr_confidence=None, word_threshold=25):
    """
    Cheap heuristic — no extra model call, no extra I/O. Runs AFTER OCR
    has already executed (ingest.py already computes ocr_text for every
    embedded image), and decides:
      - "text_dense" -> OCR text alone is sufficient (screenshots, configs, tables)
      - "diagram"    -> needs a VLM caption to capture structure/relationships
                        that OCR can't (network diagrams, flowcharts)
    """
    if not ocr_text or not ocr_text.strip():
        return "diagram"

    word_count = len(ocr_text.split())
    if word_count >= word_threshold and (ocr_confidence is None or ocr_confidence >= 0.4):
        return "text_dense"
    return "diagram"