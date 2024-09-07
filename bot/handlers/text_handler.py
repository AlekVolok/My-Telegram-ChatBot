from telegram import Update
from bot.altman.chat_api import get_openai_response

async def handle_text(update: Update, context):
    user_message = update.message.text
    response = get_openai_response(user_message)
    await update.message.reply_text(response.choices[0].message.content, parse_mode="Markdown")