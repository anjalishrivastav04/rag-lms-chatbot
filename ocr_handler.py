import easyocr
import os
from PIL import Image

# Initialize EasyOCR reader (first time will download model)
print("🔄 Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR model loaded!")

def extract_text_from_image(image_path):
    """Extract text from handwritten/scanned image using OCR."""
    print(f"🔍 Extracting text from: {os.path.basename(image_path)}")
    
    try:
        # Read image with EasyOCR
        results = reader.readtext(image_path)
        
        # Extract and combine all text
        extracted_text = "\n".join([text[1] for text in results])
        
        confidence_scores = [text[2] for text in results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        print(f"✅ Extracted {len(results)} text regions")
        print(f"📊 Average confidence: {avg_confidence:.1%}")
        
        return extracted_text, avg_confidence
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return None, 0

def save_ocr_text_to_file(image_path, output_folder="documents"):
    """Extract OCR text and save as a .txt file for LangChain parsing."""
    try:
        # Extract text using your core function above
        extracted_text, confidence = extract_text_from_image(image_path)
        
        if not extracted_text:
            return None, "Failed to extract text from image"
        
        # Generate an output filename (e.g., note_ocr.txt)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{image_name}_ocr.txt"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the raw handwritten text straight into a text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"💾 Saved OCR text to: {output_filename}")
        return output_path, None
    except Exception as e:
        return None, str(e)