from telegram import Update
from telegram.ext import ContextTypes
from bot.altman.chat_api import get_openai_response

async def handle_text(update: Update, context):
    """Handles text messages from users and sends responses from OpenAI synchronously."""
    
    user_message = update.message.text
    
    # Set the typing action while waiting for the response
    await update.message.chat.send_action(action="typing")
    
    # Get response from OpenAI (sync)
    response = get_openai_response(user_message)
    
    # Reply with the OpenAI response
    await update.message.reply_text(response.choices[0].message.content, parse_mode="Markdown")
s