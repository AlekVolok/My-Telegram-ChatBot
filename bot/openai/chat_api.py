import openai
from bot.utils.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

async def get_openai_response(prompt):
    response = await openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    return response
