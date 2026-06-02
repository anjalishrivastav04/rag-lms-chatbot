from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import glob
import os

# --- IMPORTS FOR AI TAGGING ---
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 🎛️ SWITCH PROVIDER HERE: Set to "gemini" or "grok"
PROVIDER = "gemini" 

# --- SCHEMA FOR STRUCTURED OUTPUT ---
class DocumentTags(BaseModel):
    keywords: list[str] = Field(description="List of 3-5 high-quality keywords or technical terms related to the text chunk.")

# --- GEMINI TAGGING FUNCTION ---
def get_keywords_gemini(client: genai.Client, text_content: str) -> list[str]:
    prompt = f"Analyze the following document chunk and extract the most relevant keywords or technical topics:\n\n{text_content}"
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DocumentTags,
                temperature=0.1
            ),
        )
        return DocumentTags.model_validate_json(response.text).keywords
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return []

# --- GROK TAGGING FUNCTION ---
def get_keywords_grok(client: OpenAI, text_content: str) -> list[str]:
    prompt = f"Analyze the following document chunk and extract the most relevant keywords or technical topics:\n\n{text_content}"
    try:
        response = client.beta.chat.completions.parse(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts high-quality metadata keywords from technical text."},
                {"role": "user", "content": prompt}
            ],
            response_format=DocumentTags,
            temperature=0.1
        )
        result = response.choices[0].message.parsed
        return result.keywords if result else []
    except Exception as e:
        print(f"⚠️ Grok Error: {e}")
        return []

def ingest_documents():
    docs = []

    # Load .txt files
    for path in glob.glob("documents/*.txt"):
        loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
            doc.metadata["filetype"] = "txt"
        docs.extend(loaded)

    # Load .pdf files
    for path in glob.glob("documents/*.pdf"):
        loader = PyPDFLoader(path)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
            doc.metadata["filetype"] = "pdf"
        docs.extend(loaded)

    if not docs:
        print("No documents found in /documents folder!")
        return

    # 🛠️ --- ADVANCED PARENT-CHILD SPLITTING LAYER ---
    # Parent documents capture enough layout to hold explanations together
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    # Child splitter generates sharp chunks to give highly accurate semantic searches
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    
    final_child_chunks = []
    global_chunk_counter = 0

    print("✂️ Splitting documents into parent and child structural hierarchies...")
    for doc in docs:
        # Step 1: Split source files into larger parent blocks
        parent_chunks = parent_splitter.split_documents([doc])
        
        for p_chunk in parent_chunks:
            # Step 2: Split those parent blocks into small searchable chunks
            child_chunks = child_splitter.split_documents([p_chunk])
            
            for c_chunk in child_chunks:
                # Step 3: Embed the parent content and source details inside the child's metadata
                c_chunk.metadata["parent_content"] = p_chunk.page_content
                c_chunk.metadata["source"] = doc.metadata.get("source")
                c_chunk.metadata["filetype"] = doc.metadata.get("filetype")
                c_chunk.metadata["chunk_index"] = global_chunk_counter
                final_child_chunks.append(c_chunk)
                global_chunk_counter += 1

    chunks = final_child_chunks
    # -----------------------------------------------

    # Initialize the selected client
    if PROVIDER == "gemini":
        ai_client = genai.Client()
        print(f"\n✨ Extracting keyword tags via GEMINI for {len(chunks)} chunks...")
    elif PROVIDER == "grok":
        ai_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        print(f"\n✨ Extracting keyword tags via GROK for {len(chunks)} chunks...")
    else:
        print("❌ Invalid PROVIDER selected! Choose 'gemini' or 'grok'.")
        return

    # Add AI generated keyword tags to child chunks
    for i, chunk in enumerate(chunks):
        # We pass child text content to keep structural summary crisp
        if PROVIDER == "gemini":
            ai_keywords = get_keywords_gemini(ai_client, chunk.page_content)
        elif PROVIDER == "grok":
            ai_keywords = get_keywords_grok(ai_client, chunk.page_content)
            
        chunk.metadata["keywords"] = ai_keywords
        print(f"  Processed chunk {i+1}/{len(chunks)} | Tags: {ai_keywords}")

    # Print tags for verification
    print("\n📌 Tagged Chunks Preview:")
    for chunk in chunks[:3]:
        print(f"  Source: {chunk.metadata['source']} | Type: {chunk.metadata['filetype']} | Chunk: {chunk.metadata['chunk_index']}")
        print(f"  Keywords: {chunk.metadata['keywords']}")
        print(f"  Content: {chunk.page_content[:80]}...")
        print()

    # Save vector store
    print("📦 Generating embeddings and saving FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")

    print(f"✅ Ingested {len(chunks)} chunks from structural hierarchies using {PROVIDER.upper()} tags!")

if __name__ == "__main__":
    ingest_documents()