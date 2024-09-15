from telegram import Update
from telegram.ext import ContextTypes
from bot.altman.chat_api import get_openai_response
from bot.database import sqlite_manager


async def handle_message(update: Update, context):
    user_id, folder = sqlite_manager.register_user(update.message.from_user.username)

    # Assume we track the current topic in the session or by user
    topic = sqlite_manager.get_current_topic(user_id)  # Retrieve the current topic

    # Save the user's message
    sqlite_manager.save_message(user_id, topic, update.message.text, is_bot_response=False)

    # Get response from OpenAI API
    openai_response = get_openai_response(update.message.text, user_id, topic)

    # Save the bot's response
    sqlite_manager.save_message(user_id, topic, openai_response, is_bot_response=True)

    # Send the response back to the user
    await update.message.reply_text(openai_response)
