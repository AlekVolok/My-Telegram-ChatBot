from openai import OpenAI
client = OpenAI()

from bot.utils.config import OPENAI_API_KEY

client.api_key = OPENAI_API_KEY

def get_openai_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    return response
