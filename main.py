import os
import time
import tempfile
import asyncio

from telegram import (
    Update, 
    BotCommand,
    BotCommandScopeDefault,  
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
    filters,
    ConversationHandler
)
from telegram.constants import BotCommandScopeType
from dotenv import load_dotenv
import tiktoken
import docx
from PyPDF2 import PdfReader
import aiosqlite  # Use aiosqlite for async database operations
from openai import OpenAI  # Retained OpenAI import as per user's instruction
import google.generativeai as genai  # Import Gemini's SDK

### CONFIG ###
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Add Gemini API Key

MODELS_INFO = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Lightweight GPT-4 model",
        "scores": {"speed": 5, "accuracy": 3},
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "max_tokens": 100000  # Per-model max_tokens
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Great for most tasks",
        "scores": {"speed": 3, "accuracy": 5},
        "provider": "openai",
        "api_model": "gpt-4o",
        "max_tokens": 100000  # Per-model max_tokens
    },
    "gemini": {
        "name": "Gemini",
        "description": "Google's new Gemini AI. Multi-modal capabilities. 1M token limit!",
        "scores": {"speed": 4, "accuracy": 4},
        "provider": "gemini",
        "api_model": "gemini-1.5-flash",
        "max_tokens": 900000  # Per-model max_tokens
    },
    "bing": {
        "name": "Bing",
        "description": "GPT-4 with Bing internet search",
        "scores": {"speed": 5, "accuracy": 2},
        "provider": "bing",
        "api_model": "bing",
        "max_tokens": 100000  # Per-model max_tokens
    }
}

### DATABASE ###
DATABASE_PATH = "chatbot.db"
ENCODER = tiktoken.encoding_for_model("gpt-4")
# MAX_TOKENS = 100000  # Removed global MAX_TOKENS as we now have per-model max_tokens
DEFAULT_MODEL = "gpt-4o-mini"

# Conversation states
TOPIC_INPUT = 1

# Function to initialize database and tables
async def init_db(application):
    """Initialize the database and create necessary tables."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()

        # Create table to store users
        await c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,  -- Telegram user ID
                username TEXT,
                current_model TEXT DEFAULT 'gpt-4o-mini',  -- Default model
                current_topic TEXT  -- Current topic
            )
        ''')
        print("Users table created or already exists.")

        # Create table to store topics
        await c.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                topic TEXT,
                deleted INTEGER DEFAULT 0,  -- 0 = Active, 1 = Deleted
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        print("Topics table created or already exists.")

        # Create table to store chat history
        await c.execute('''
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
        print("Chat history table created or already exists.")

        await conn.commit()


# Function to migrate database (add 'deleted' column if not exists)
async def migrate_db(application):
    """Migrate the database schema if necessary."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()

        # Check if 'topics' table exists
        await c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topics';")
        table_exists = await c.fetchone()

        if table_exists:
            # Check if 'deleted' column exists in 'topics' table
            await c.execute("PRAGMA table_info(topics)")
            columns = await c.fetchall()
            column_names = [column[1] for column in columns]
            if 'deleted' not in column_names:
                await c.execute("ALTER TABLE topics ADD COLUMN deleted INTEGER DEFAULT 0")
                await conn.commit()
        else:
            # The 'topics' table does not exist; skip migration
            print("Topics table does not exist. Skipping migration.")

# Function to register a user
async def register_user(user_id, username):
    """Register a new user or update existing user."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT id, current_model, current_topic FROM users WHERE id=?", (user_id,))
        result = await c.fetchone()

        if result:
            # If current_model is None, set it to the default
            if result[1] is None:
                await c.execute("UPDATE users SET current_model = ? WHERE id = ?", (DEFAULT_MODEL, user_id))
                await conn.commit()
        else:
            # Insert user with default model and no current topic
            await c.execute("INSERT INTO users (id, username, current_model, current_topic) VALUES (?, ?, ?, ?)", 
                          (user_id, username, DEFAULT_MODEL, None))
            await conn.commit()

# Function to start a new chat
async def start_new_chat(user_id, topic=None):
    """Start a new chat for the user with the specified topic."""
    if not topic:
        topic = f'chat_{user_id}_{int(time.time())}'

    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        # Check if topic exists and is not deleted
        await c.execute("SELECT topic FROM topics WHERE user_id=? AND topic=? AND deleted=0", (user_id, topic))
        result = await c.fetchone()
        if not result:
            await c.execute("INSERT INTO topics (user_id, topic) VALUES (?, ?)", (user_id, topic))
            await conn.commit()

        # Set current topic
        await set_current_topic(user_id, topic)

        # Log new chat
        await save_message(user_id, topic, "New chat started", is_bot_response=True)

    return topic

# Function to save a message
async def save_message(user_id, topic, message, is_bot_response):
    """Save a message to the chat history."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute(
            "INSERT INTO chat_history (user_id, topic, message, is_bot_response) VALUES (?, ?, ?, ?)",
            (user_id, topic, message, int(is_bot_response))
        )
        await conn.commit()

# Function to count tokens
def count_tokens(messages):
    """
    Counts the total number of tokens for the given messages.
    Each message is expected to have 'role' and 'content' fields.
    """
    total_tokens = 0
    for message in messages:
        total_tokens += count_text_tokens(message['content']) + 4  # 4 tokens per message for roles/metadata
    return total_tokens

def count_text_tokens(text: str):
    """Counts the number of tokens in a given text."""
    return len(ENCODER.encode(text))

# Function to get conversation history
async def get_conversation_history(user_id, topic, max_tokens):
    """Retrieve and format the conversation history for the user and topic."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        # Check if the topic is not deleted
        await c.execute(
            "SELECT deleted FROM topics WHERE user_id=? AND topic=?",
            (user_id, topic)
        )
        topic_status = await c.fetchone()
        if not topic_status or topic_status[0] == 1:
            # Topic is deleted or does not exist
            return []

        await c.execute(
            "SELECT message, is_bot_response FROM chat_history WHERE user_id=? AND topic=? ORDER BY created_at ASC",
            (user_id, topic)
        )
        messages = await c.fetchall()

    # Format messages for AI API
    formatted_messages = []
    for msg, is_bot in messages:
        role = 'assistant' if is_bot else 'user'
        formatted_messages.append({"role": role, "content": msg})

    # Ensure the token count does not exceed the model's max_tokens
    total_tokens = count_tokens(formatted_messages)
    while total_tokens > max_tokens and formatted_messages:
        # Remove the oldest message
        formatted_messages.pop(0)
        total_tokens = count_tokens(formatted_messages)

    return formatted_messages

# Function to get current topic
async def get_current_topic(user_id):
    """Retrieve the current topic for the user."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute(
            "SELECT current_topic FROM users WHERE id=?",
            (user_id,)
        )
        topic = await c.fetchone()

    if topic and topic[0]:
        # Verify that the topic is not deleted
        async with aiosqlite.connect(DATABASE_PATH) as conn:
            c = await conn.cursor()
            await c.execute(
                "SELECT deleted FROM topics WHERE user_id=? AND topic=?",
                (user_id, topic[0])
            )
            result = await c.fetchone()
            if result and result[0] == 0:
                return topic[0]
            else:
                # Current topic is deleted; unset it
                await set_current_topic(user_id, None)
                return None
    else:
        # No current topic
        return None

# Function to get user model
async def get_user_model(user_id):
    """Retrieve the current AI model for the user."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT current_model FROM users WHERE id=?", (user_id,))
        result = await c.fetchone()

    if result and result[0]:
        return result[0]  # Return the model key (e.g., 'gpt-4o-mini')
    else:
        await set_user_model(user_id, DEFAULT_MODEL)
    return DEFAULT_MODEL

# Function to set user model
async def set_user_model(user_id, model_key):
    """Set the AI model for the user."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as conn:
            c = await conn.cursor()
            await c.execute("UPDATE users SET current_model=? WHERE id=?", (model_key, user_id))
            await conn.commit()
    except aiosqlite.Error as e:
        print(f"Error setting model {model_key} for user ID {user_id}: {e}")

# Function to set current topic
async def set_current_topic(user_id, topic):
    """Set the current topic for the user."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as conn:
            c = await conn.cursor()
            await c.execute("UPDATE users SET current_topic=? WHERE id=?", (topic, user_id))
            await conn.commit()
    except aiosqlite.Error as e:
        print(f"Error setting current topic for user ID {user_id}: {e}")

### OPENAI & GEMINI CONFIGURATION ###
# Configure OpenAI
client = OpenAI()
client.api_key = OPENAI_API_KEY  # Retained OpenAI configuration as per user's instruction

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)  # Added Gemini configuration

### RESPONSE GENERATION ###
async def get_ai_response(prompt: str, user_id, topic):
    """Generates a response using the selected AI model."""
    selected_model = await get_user_model(user_id)
    max_tokens = MODELS_INFO[selected_model]['max_tokens']  # Retrieve per-model max_tokens
    conversation_history = await get_conversation_history(user_id, topic, max_tokens)
    conversation_history.append({"role": "user", "content": prompt})

    provider = MODELS_INFO[selected_model]["provider"]
    api_model = MODELS_INFO[selected_model]["api_model"]

    try:
        if provider == "openai":
            # OpenAI's response generation remains unchanged
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=api_model,
                messages=conversation_history
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "gemini":
            # Using Gemini's API for response generation
            model = genai.GenerativeModel(api_model)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt
            )
            return response.text.strip()
        
        elif provider == "bing":
            # Placeholder for Bing integration
            # You need to implement Bing's API similar to OpenAI and Gemini
            return "Bing integration is not yet implemented."
        
        else:
            return "Selected AI provider is not supported."
    
    except Exception as e:
        print(f"Error generating response with {provider}: {e}")
        return "Sorry, I couldn't process your request at the moment."

### HELPERS ###
# Helper function to extract text from a file
def extract_text_from_file(file_path: str, mime_type: str) -> str:
    if mime_type == 'application/pdf':
        # Extract text from PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
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

# Handler for text messages
async def handle_text(update: Update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)

    topic = await get_current_topic(user_id)
    if not topic:
        await update.message.reply_text("Please start a new chat using /new command.")
        return

    # Set the typing action while waiting for the response
    await update.message.chat.send_action(action="typing")

    await save_message(user_id, topic, update.message.text, is_bot_response=False) 

    ai_response = await get_ai_response(update.message.text, user_id, topic)
    await save_message(user_id, topic, ai_response, is_bot_response=True)

    # Reply with the AI response
    await update.message.reply_text(ai_response, parse_mode="Markdown")

# Handler for document messages
async def handle_document(update: Update, context):
    user_message = update.message.caption if update.message.caption else " "  # Default message if no text is provided
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
        
        # Retrieve the user's selected model and its max_tokens
        user_id = update.message.from_user.id
        selected_model = await get_user_model(user_id)
        max_tokens = MODELS_INFO[selected_model]['max_tokens']  # Retrieve per-model max_tokens

        file_tokens = count_text_tokens(extracted_text)
        if file_tokens > max_tokens:
            await update.message.reply_text(f"Sorry, the document is too large ({file_tokens} tokens) to process. The maximum allowed token count for {MODELS_INFO[selected_model]['name']} is {max_tokens}.")
            os.remove(temp_file_path)
            return

        final_text = user_message + ":   " + extracted_text

        # Set the typing action while waiting for the response
        await update.message.chat.send_action(action="typing")

        # Database operations
        username = update.message.from_user.username
        await register_user(user_id, username)
        topic = await get_current_topic(user_id)

        if not topic:
            await update.message.reply_text("Please start a new chat using /new command.")
            return

        # Get response from AI 
        ai_response = await get_ai_response(final_text, user_id, topic)
        await save_message(user_id, topic, ai_response, is_bot_response=True)
        
        # Reply with the AI response
        await update.message.reply_text(ai_response, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

### BOT INTERFACE ###

# Handler for /start command
async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to the bot! Use /help to see available commands.")

# Handler for /new command to start a new chat
async def start_new(update: Update, context) -> int:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)

    await update.message.reply_text("Please enter a topic name for the new chat, or type 'auto' to generate one automatically.")
    return TOPIC_INPUT

# Handler to receive topic name
async def receive_topic_name(update: Update, context) -> int:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    topic_name = update.message.text.strip()

    if topic_name.lower() == 'auto':
        topic = await start_new_chat(user_id)
        if topic:
            await update.message.reply_text(f"New chat started with topic: {topic}")
        else:
            await update.message.reply_text("Failed to start a new chat. Please try again.")
    else:
        topic = topic_name
        new_topic = await start_new_chat(user_id, topic)
        if new_topic:
            await update.message.reply_text(f"New chat started with topic: {new_topic}")
        else:
            await update.message.reply_text("Failed to start a new chat. Please try again.")

    return ConversationHandler.END

# Handler for /help command
async def help_handle(update: Update, context) -> None:
    user_id = update.message.from_user.id
    current_model = await get_user_model(user_id)
    current_topic = await get_current_topic(user_id)
    await update.message.reply_text(
        f"**Current Model:** {MODELS_INFO[current_model]['name']}"
        f"\n\n**Topic:** {current_topic}"
        "\n\n**Commands:**"
        "\n/new - Start new dialog"
        "\n/topic - Manage your topics"
        "\n/models - Manage your AI models"
        "\n/help - Show help message",
        parse_mode="Markdown"
    )

# Handler for /topic command
async def show_topic_menu(update: Update, context) -> None:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)

    text, reply_markup = await get_topic_menu_async(user_id)

    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Function to generate topic menu
async def get_topic_menu_async(user_id):
    text = "**Your Topics:**\n"

    # Fetch active topics
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT topic FROM topics WHERE user_id=? AND deleted=0", (user_id,))
        topics = await c.fetchall()

    if not topics:
        text += "You have no topics. Use /new to create one."
        buttons = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_topics")]
        ]
    else:
        buttons = []
        for topic_tuple in topics:
            topic_name = topic_tuple[0]
            buttons.append(
                [InlineKeyboardButton(topic_name, callback_data=f"select_topic|{topic_name}")]
            )
        # Add buttons for deleting topics and adding a new topic
        buttons.append([
            InlineKeyboardButton("🗑 Delete Topic", callback_data="delete_topic_menu"),
            InlineKeyboardButton("➕ Add Topic", callback_data="add_topic")
        ])

    reply_markup = InlineKeyboardMarkup(buttons)
    return text, reply_markup

# Handler for /models command
async def show_models_menu(update: Update, context) -> None:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)

    text, reply_markup = await get_models_menu_async(user_id)

    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Function to generate models menu
async def get_models_menu_async(user_id):
    current_model = await get_user_model(user_id)

    text = f"**Current Model:** {MODELS_INFO[current_model]['name']}\n\n"
    text += f"{MODELS_INFO[current_model]['description']}\n\n"
    text += "**Model Scores:**\n"
    score_dict = MODELS_INFO[current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key.capitalize()}\n"

    text += "\n**Select Model:**"

    # Buttons for models
    model_buttons = []
    for model_key in MODELS_INFO:
        title = MODELS_INFO[model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        model_buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_model|{model_key}")
        )

    reply_markup = InlineKeyboardMarkup([
        model_buttons
    ])

    return text, reply_markup

# Handler for /settings command (optional, can be removed or kept as a summary)
async def settings_handle(update: Update, context) -> None:
    # Optionally, redirect to /help or inform users to use /topic and /models
    await update.message.reply_text("Please use /topic to manage your topics and /models to manage your AI models.")

# Callback Query Handlers

# Handler to set model
async def set_model_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Get selected model from callback data
    _, model_key = query.data.split("|")

    # Validate model_key
    if model_key not in MODELS_INFO:
        await query.answer("Invalid model selected.")
        return

    # Set the user's selected model in the database
    await set_user_model(user_id, model_key)

    # Get updated models menu
    text, reply_markup = await get_models_menu_async(user_id)

    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    except error.BadRequest as e:
        if "Message is not modified" not in str(e):
            print(f"Error editing message: {e}")

# Handler to select topic
async def select_topic_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Get selected topic from callback data
    _, topic_name = query.data.split("|")

    # Check if the topic exists for the user and is not deleted
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT topic FROM topics WHERE user_id=? AND topic=? AND deleted=0", (user_id, topic_name))
        result = await c.fetchone()

    if result:
        # Set the user's current topic
        await set_current_topic(user_id, topic_name)

        # Send confirmation
        await query.edit_message_text(f"**Current topic set to:** {topic_name}", parse_mode="Markdown")
    else:
        await query.answer("Topic not found or has been deleted.")

# Handler to delete topic menu
async def delete_topic_menu_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Get the list of active topics
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT topic FROM topics WHERE user_id=? AND deleted=0", (user_id,))
        topics = await c.fetchall()

    if not topics:
        await query.edit_message_text("You have no topics to delete.")
        return

    # Build buttons for deleting topics
    buttons = []
    for (topic_name,) in topics:
        buttons.append(
            [InlineKeyboardButton("🗑 " + topic_name, callback_data=f"delete_topic|{topic_name}")]
        )

    # Add a back button
    buttons.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_topic_menu")])

    reply_markup = InlineKeyboardMarkup(buttons)

    await query.edit_message_text("Select a topic to delete:", reply_markup=reply_markup)

# Handler to delete a specific topic
async def delete_topic_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Get selected topic from callback data
    _, topic_name = query.data.split("|")

    # Mark the topic as deleted
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("UPDATE topics SET deleted=1 WHERE user_id=? AND topic=?", (user_id, topic_name))
        await conn.commit()

    # If the deleted topic was the current topic, unset it
    current_topic = await get_current_topic(user_id)
    if current_topic == topic_name:
        await set_current_topic(user_id, None)

    # Confirm deletion
    await query.edit_message_text(f"Topic '{topic_name}' has been deleted.", parse_mode="Markdown")

# Handler to add a new topic
async def add_topic_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Prompt user to enter a new topic name
    await query.edit_message_text("Please enter the name of the new topic:")
    return TOPIC_INPUT

# Handler to refresh topics
async def refresh_topics_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    text, reply_markup = await get_topic_menu_async(user_id)

    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Handler to go back to topic menu
async def back_to_topic_menu_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    text, reply_markup = await get_topic_menu_async(user_id)

    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Handler to go back to models menu
async def back_to_models_menu_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    text, reply_markup = await get_models_menu_async(user_id)

    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Fallback handler for ConversationHandler
async def cancel_conversation(update: Update, context):
    await update.message.reply_text("Operation cancelled.", parse_mode="Markdown")
    return ConversationHandler.END

# Function to set bot commands
async def set_bot_commands(application):
    bot_commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help message"),
        BotCommand("new", "Start new dialog"),
        BotCommand("topic", "Manage your topics"),
        BotCommand("models", "Manage your AI models"),
        BotCommand("settings", "Show settings"),
        # Add any other commands you have
    ]
    await application.bot.set_my_commands(bot_commands, scope=BotCommandScopeDefault())
    print("Bot commands have been set.")

### MAIN ###
if __name__ == '__main__':
    # Create the bot application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(init_db)
        .post_init(migrate_db)
        .post_init(set_bot_commands)
        .build()
    )

    # Conversation handler for /new command
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('new', start_new)],
        states={
            TOPIC_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_topic_name)],
        },
        fallbacks=[CommandHandler('cancel', cancel_conversation)],
    )

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_handle))
    app.add_handler(CommandHandler("topic", show_topic_menu))
    app.add_handler(CommandHandler("models", show_models_menu))
    app.add_handler(CommandHandler("settings", settings_handle))  # Optional

    # Callback handlers
    app.add_handler(CallbackQueryHandler(set_model_handle, pattern="^set_model\|"))
    app.add_handler(CallbackQueryHandler(select_topic_handle, pattern="^select_topic\|"))
    app.add_handler(CallbackQueryHandler(delete_topic_menu_handle, pattern="^delete_topic_menu"))
    app.add_handler(CallbackQueryHandler(delete_topic_handle, pattern="^delete_topic\|"))
    app.add_handler(CallbackQueryHandler(add_topic_handle, pattern="^add_topic"))
    app.add_handler(CallbackQueryHandler(refresh_topics_handle, pattern="^refresh_topics"))
    app.add_handler(CallbackQueryHandler(back_to_topic_menu_handle, pattern="^back_to_topic_menu"))
    app.add_handler(CallbackQueryHandler(back_to_models_menu_handle, pattern="^back_to_models_menu"))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Start the bot (this will run the event loop)
    app.run_polling()
