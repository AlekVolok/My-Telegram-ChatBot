from telegram import Update

async def handle_text(update: Update, context):
    await update.message.reply_text(f"You said: {update.message.text}")
