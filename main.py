from telegram import (
    Update, 
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    error
    )
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler,
    CallbackQueryHandler, 
    CallbackContext,
    ContextTypes,
    filters
    )
from dotenv import load_dotenv
import os

import pdb

### CONFIG ###
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODELS_INFO = {
    "gpt-4o-mini": {"name": "GPT-4o Mini", "description": "Lightweight GPT-4 model", "scores": {"speed": 5, "accuracy": 3}},
    "gpt-4o": {"name": "GPT-4", "description": "Great for most tasks", "scores": {"speed": 3, "accuracy": 5}},
    "gemini": {"name": "Gemini", "description": "Google's new Gemini AI", "scores": {"speed": 4, "accuracy": 4}},
    "bing": {"name": "Bing", "description": "Bing AI", "scores": {"speed": 5, "accuracy": 2}}
}

### DATABASE ###
import sqlite3
import time
import tiktoken

DATABASE_PATH = "chatbot.db"
encoding = tiktoken.get_encoding("cl100k_base")
ENCODER = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS = 100000  # Maximum allowed token count
DEFAULT_MODEL = "gpt-4o-mini"

# Function to initialize database and tables
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Create table to store users
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,  -- Telegram user ID
            username TEXT,
            current_model TEXT DEFAULT 'gpt-4o-mini'  -- Default model
        )
    ''')

    # Create table to store chat history
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            topic TEXT,
            message TEXT,
            is_bot_response INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()


def register_user(user_id, username):
    """Register a new user or return the existing user's ID."""
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, current_model FROM users WHERE id=?", (user_id,))
            result = c.fetchone()

            if result:
                # If current_model is None, set it to the default
                if result[1] is None:
                    c.execute("UPDATE users SET current_model = ? WHERE id = ?", (DEFAULT_MODEL, user_id))
                    conn.commit()
            else:
                # Insert user with default model
                c.execute("INSERT INTO users (id, username, current_model) VALUES (?, ?, ?)", (user_id, username, DEFAULT_MODEL))
                conn.commit()

        return user_id
    except sqlite3.Error as e:
        print(f"Error registering user {username}: {e}")
        return None


def start_new_chat(user_id):
    topic = f'chat_{user_id}_{int(time.time())}'
    save_message(user_id, topic, "New chat started", is_bot_response=True)  # Log this new chat

    return topic


def save_message(user_id, topic, message, is_bot_response):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Insert the message into the chat_history
    c.execute(
        "INSERT INTO chat_history (user_id, topic, message, is_bot_response) VALUES (?, ?, ?, ?)",
        (user_id, topic, message, is_bot_response)
    )

    conn.commit()
    conn.close()


def count_tokens(messages):
    """
    Counts the total number of tokens for the given messages.
    Each message is expected to have 'role' and 'content' fields.
    """
    total_tokens = 0
    for message in messages:
        total_tokens += count_text_tokens(message['content']) + 4  # 4 tokens per message for roles/metadata
    return total_tokens


def count_text_tokens(text:str):
    return len(ENCODER.encode(text))


def get_conversation_history(user_id, topic):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT message, is_bot_response FROM chat_history WHERE user_id=? AND topic=? ORDER BY created_at ASC",
        (user_id, topic)
    )

    messages = c.fetchall()
    conn.close()

    # Format messages for OpenAI API
    formatted_messages = []
    for msg, is_bot in messages:
        role = 'assistant' if is_bot else 'user'
        formatted_messages.append({"role": role, "content": msg})

    # Ensure the token count does not exceed the limit
    total_tokens = count_tokens(formatted_messages)
    while total_tokens > MAX_TOKENS:
        # Remove the oldest message
        formatted_messages.pop(0)
        total_tokens = count_tokens(formatted_messages)

    return formatted_messages


def get_current_topic(user_id):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT topic FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
        (user_id,)
    )

    topic = c.fetchone()
    conn.close()

    if topic:
        return topic[0]
    else:
        # Start a new chat if there is no topic
        return start_new_chat(user_id)


def ensure_database_path():
    directory = os.path.dirname(DATABASE_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_user_model(user_id):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    c.execute("SELECT current_model FROM users WHERE id=?", (user_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return result[0]  # Return the model (e.g., 'gpt-4o-mini')
    else:
        set_user_model(user_id, DEFAULT_MODEL)
    return DEFAULT_MODEL


def set_user_model(user_id, model_key):
    """Set the model for the user."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("UPDATE users SET current_model=? WHERE id=?", (model_key, user_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error setting model {model_key} for user ID {user_id}: {e}")



### OPENAI ###
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Set the OpenAI API key
client.api_key = OPENAI_API_KEY

def get_openai_response(prompt: str, user_id, topic):
    """Sends a text prompt to OpenAI and returns the response."""
    conversation_history = get_conversation_history(user_id, topic)
    conversation_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=get_user_model(user_id),
        messages=conversation_history
    )
    
    return response.choices[0].message.content


### HELPERS ###
import docx
from PyPDF2 import PdfReader
import tempfile

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

### HANDLERS ###

async def handle_text(update: Update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    register_user(user_id, username)
    
    # Set the typing action while waiting for the response
    await update.message.chat.send_action(action="typing")
    
    topic = get_current_topic(user_id)
    save_message(user_id, topic, update.message.text, is_bot_response=False) 
    
    openai_response = get_openai_response(update.message.text, user_id, topic)
    save_message(user_id, topic, openai_response, is_bot_response=True)
    
    # Reply with the OpenAI response
    await update.message.reply_text(openai_response, parse_mode="Markdown")


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
            os.remove(temp_file_path)
            return

        # Send an initial message to the user to indicate that processing has started
        msg = await update.message.reply_text("Processing the document...")
        
        file_tokens = count_text_tokens(extracted_text)
        if file_tokens > MAX_TOKENS:
            await update.message.reply_text(f"Sorry, the document is too large ({file_tokens} tokens) to process. The maximum allowed token count is {MAX_TOKENS}.")
            os.remove(temp_file_path)
            return
        
        final_text = user_message + ":   " + extracted_text
        
        # Set the typing action while waiting for the response
        await update.message.chat.send_action(action="typing")
        
        # Database operations
        user_id = update.message.from_user.id
        username = update.message.from_user.username
        register_user(user_id, username)
        topic = get_current_topic(user_id)

        # Get response from OpenAI 
        openai_response = get_openai_response(final_text, user_id, topic)
        save_message(user_id, topic, openai_response, is_bot_response=True)
        
        # Reply with the OpenAI response
        await update.message.reply_text(openai_response, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)



### BOT INTERFACE ###
async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to the bot!")


async def start_new(update: Update, context) -> None:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    register_user(user_id, username)
    topic = start_new_chat(user_id)
    await update.message.reply_text(f"New chat started")


async def help_handle(update: Update, context) -> None:
    user_id = update.message.from_user.id
    current_model = get_user_model(user_id)
    current_topic = get_current_topic(user_id)
    await update.message.reply_text(
        f"Current Model: {current_model}"
        f"\n\nTopic: {current_topic}"
        "\n\n/new - Start new dialog"
        "\n/settings - Show settings"
        "\n/topic - Show current topic"
        "\n/help - Show help message"
    )


async def show_topic(update: Update, context) -> None:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    register_user(user_id, username)
    topic = get_current_topic(user_id)
    await update.message.reply_text(f"Current topic: {topic}")


async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/settings", "Start the bot"),
        BotCommand("/topic", "Show current topic"),
        BotCommand("/help", "Show help message"),
    ])


def get_settings_menu(user_id):
    current_model = get_user_model(user_id)

    text = MODELS_INFO[current_model]["description"]

    text += "\n\n"
    score_dict = MODELS_INFO[current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in MODELS_INFO:
        title = MODELS_INFO[model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    register_user(user_id, username)
    
    text, reply_markup = get_settings_menu(user_id)

    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def set_settings_handle(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    username = query.from_user.username
    register_user(user_id, username)
    
    # Get selected model from callback data
    _, model_key = query.data.split("|")

    # Set the user's selected model in the database
    set_user_model(user_id, model_key)

    # Get updated settings menu
    text, reply_markup = get_settings_menu(user_id)

    await query.answer()

    try:
        # Check if the message content or reply markup has changed
        if query.message.text != text or query.message.reply_markup != reply_markup:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
        else:
            print("Message content and reply markup are the same, no need to edit.")
    except error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass  # Ignore this specific error
        else:
            print(f"Error editing message: {e}")


### MAIN ###
if __name__ == '__main__':
    # Initialize the SQLite database
    init_db()

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
    app.add_handler(CommandHandler("settings", settings_handle))
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.TEXT, handle_document))
    
    app.run_polling()
    pdb.set_trace()
