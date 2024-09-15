import os
import tempfile
from telegram import Update
from telegram.ext import ContextTypes
from bot.altman.chat_api import get_openai_response
from PyPDF2 import PdfReader
import docx
from bot.database import sqlite_manager

# Helper function to extract text from a file
def extract_text_from_file(file_path: str, mime_type: str) -> str:
    if mime_type == 'application/pdf':
        # Extract text from PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Extract text from DOCX file
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif mime_type == 'text/markdown' or file_path.endswith('.md'):
        # Read markdown files as plain text
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        # Assume it's a plain text file for simplicity
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    user_message = update.message.caption if update.message.caption else " " # Default message if no text is provided
    # Download the document from Telegram
    document = update.message.document
    file = await document.get_file()


    # Save the document to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        await file.download_to_drive(temp_file_path)

    try:
        # Extract text from the document based on its mime type
        extracted_text = extract_text_from_file(temp_file_path, document.mime_type)

        if not extracted_text.strip():
            await update.message.reply_text("Sorry, I couldn't extract any text from this document.")
            return

        # Send an initial message to the user to indicate that processing has started
        msg = await update.message.reply_text("Processing the document...")

        final_text = user_message + ":   " + extracted_text
        
        # Set the typing action while waiting for the response
        await update.message.chat.send_action(action="typing")
        
        # Database operations
        user_id = sqlite_manager.register_user(update.message.from_user.username)
        topic = sqlite_manager.get_current_topic(user_id)

        # Get response from OpenAI 
        openai_response = get_openai_response(final_text, user_id, topic)
        sqlite_manager.save_message(user_id, topic, openai_response, is_bot_response=True)
        
        # Reply with the OpenAI response
        await update.message.reply_text(openai_response, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
