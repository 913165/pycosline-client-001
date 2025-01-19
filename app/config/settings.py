# app/config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
VECTOR_DB_URL = "http://localhost:7272"
ALLOWED_ORIGINS = ["http://localhost:3000"]