from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from bot.handlers.text_handler import handle_text
from bot.utils.config import TELEGRAM_TOKEN

async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to the bot!")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()
