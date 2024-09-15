from openai import OpenAI
from bot.utils.config import OPENAI_API_KEY
from bot.database import sqlite_manager

# Initialize the OpenAI client
client = OpenAI()

# Set the OpenAI API key
client.api_key = OPENAI_API_KEY

def get_openai_response(prompt: str, user_id, topic):
    """Sends a text prompt to OpenAI and returns the response."""
    conversation_history = sqlite_manager.get_conversation_history(user_id, topic)
    conversation_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Replace with the model you're using
        messages=conversation_history
    )
    
    return response.choices[0].message.content
