from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler,
    CallbackQueryHandler, 
    filters
    )
from bot.handlers.text_handler import handle_text
from bot.handlers.document_handler import handle_document
import bot.handlers.interface as bot_interface
from bot.utils.config import TELEGRAM_TOKEN
from bot.database import sqlite_manager

import pdb


if __name__ == '__main__':
    # Initialize the SQLite database
    sqlite_manager.init_db()

    # Create the bot application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(bot_interface.post_init)
        .build()
    )

    # Command handlers
    app.add_handler(CommandHandler("start", bot_interface.start))
    app.add_handler(CommandHandler("new", bot_interface.start_new))
    app.add_handler(CommandHandler("help", bot_interface.help_handle))
    app.add_handler(CommandHandler("topic", bot_interface.show_topic))
    app.add_handler(CommandHandler("settings", bot_interface.settings_handle))
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(bot_interface.set_settings_handle, pattern="^set_settings"))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.TEXT, handle_document))
    
    app.run_polling()
    pdb.set_trace()
