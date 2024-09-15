from telegram import Update
from telegram.ext import ContextTypes
from bot.database import sqlite_manager
from bot.altman.chat_api import get_openai_response

async def handle_text(update: Update, context):

    user_id = sqlite_manager.register_user(update.message.from_user.username)
    
    # Set the typing action while waiting for the response
    await update.message.chat.send_action(action="typing")
    
    topic = sqlite_manager.get_current_topic(user_id)
    sqlite_manager.save_message(user_id, topic, update.message.text, is_bot_response=False) 
    
    openai_response = get_openai_response(update.message.text, user_id, topic)
    sqlite_manager.save_message(user_id, topic, openai_response, is_bot_response=True)
    
    # Reply with the OpenAI response
    await update.message.reply_text(openai_response, parse_mode="Markdown")
