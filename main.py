from telegram import Update, BotCommand
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


async def help_handle(update: Update, context) -> None:
    await update.message.reply_text(
        "Commands:" 
        "\n/new - Start new dialog"
        "\n/start - Start the bot"
        "\n/topic - Show current topic"
        "\n/help - Show help message"
    )


async def show_topic(update: Update, context) -> None:
    user_id = sqlite_manager.register_user(update.message.from_user.username)
    topic = sqlite_manager.get_current_topic(user_id)
    await update.message.reply_text(f"Current topic: {topic}")


async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/start", "Start the bot"),
        BotCommand("/topic", "Show current topic"),
        BotCommand("/help", "Show help message"),
    ])


if __name__ == '__main__':
    # Initialize the SQLite database
    sqlite_manager.init_db()

    # Create the bot application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", start_new))
    app.add_handler(CommandHandler("help", help_handle))
    app.add_handler(CommandHandler("topic", show_topic))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.TEXT, handle_document))
    
    app.run_polling()
    pdb.set_trace()
