import os
from dotenv import load_dotenv
import cohere
from langfuse.openai import openai

# Load .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path, override=True)

# Retrieve API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_API_KEY = os.getenv("LANGFUSE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("⚠️ Cohere API key not found in .env file!")

# Initialize API Clients
co = cohere.Client(COHERE_API_KEY)
client = openai.OpenAI()
