from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from bot.handlers.text_handler import handle_text
from bot.handlers.document_handler import handle_document
from bot.utils.config import TELEGRAM_TOKEN
from bot.database import sqlite_manager

import pdb

async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to the bot!")

async def start_new(update: Update, context) -> None:
    user_id = sqlite_manager.register_user(update.message.from_user.username)
    topic = sqlite_manager.start_new_chat(user_id)
    await update.message.reply_text(f"New chat started")

if __name__ == '__main__':
    # Initialize the SQLite database
    sqlite_manager.init_db()

    # Create the bot application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("newchat", start_new))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.TEXT, handle_document))
    
    app.run_polling()
    pdb.set_trace()
