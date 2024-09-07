import os
from dotenv import load_dotenv

load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_CHATBOT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_CHATBOT")


