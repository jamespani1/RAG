
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "uploads")
VECTOR_STORE_DIRECTORY = os.path.join(BASE_DIR, "vector_store")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_API_KEY = "api_key"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100