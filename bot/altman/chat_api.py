from openai import OpenAI
from bot.utils.config import OPENAI_API_KEY

# Initialize the OpenAI client
client = OpenAI()

# Set the OpenAI API key
client.api_key = OPENAI_API_KEY

def get_openai_response(prompt: str):
    """Sends a text prompt to OpenAI and returns the response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Replace with the model you're using
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response
