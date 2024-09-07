from openai import OpenAI
client = OpenAI()

from bot.utils.config import OPENAI_API_KEY

client.api_key = OPENAI_API_KEY

async def get_openai_response(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ],
        stream=True
    )
    return response
