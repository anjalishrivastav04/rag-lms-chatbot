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