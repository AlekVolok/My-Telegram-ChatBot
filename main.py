import os
import re
import time
import tempfile
import asyncio

from telegram import (
    Update, 
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    helpers,
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
COST_INTEREST_RATE = 1.2  # 20% interest rate for billing

MODELS_INFO = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Lightweight GPT-4 model",
        "scores": {
            "Chat": 1273,
            "Coding": 1282
            },
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "max_tokens": 100000,  # Per-model max_tokens
        "cost_input": 0.15,  # Cost per 1M token for billing
        "cost_output": 0.6  
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Great for most tasks",
        "scores": {
            "Chat": 1335,
            "Coding": 1341
            },
        "provider": "openai",
        "api_model": "gpt-4o",
        "max_tokens": 100000,  # Per-model max_tokens
        "cost_input": 2.5,  # Cost per 1M token for billing
        "cost_output": 10.0  
    },
    "gemini": {
        "name": "Gemini 1.5 Flash",
        "description": "Google's latest AI. Features:\n -Image processing\n -Audio processing",
        "scores": {
            "Chat": 1269,
            "Coding": 1257
            },
        "provider": "gemini",
        "api_model": "gemini-1.5-flash",
        "max_tokens": 100000,  # Per-model max_tokens. Reduced because of Free Tier limit
        "cost_input": 0.075,  # Cost per 1M token for billing
        "cost_output": 0.3  # Cost per 1M token for billing
    }
    # "grok": {
    #     "name": "Grok-2",
    #     "description": "Elon Musk's latest AI",
    #     "scores": {
    #         "Chat": 1294,
    #         "Coding": 1286
    #         },
    #     "provider": "bing",
    #     "api_model": "bing",
    #     "max_tokens": 100000,
    #     "cost_input": 0.1,
    #     "cost_output": 0.4
    # }
}

### DATABASE ###
DATABASE_PATH = "chatbot.db"
ENCODER = tiktoken.encoding_for_model("gpt-4")
# MAX_TOKENS = 100000  # Removed global MAX_TOKENS as we now have per-model max_tokens
DEFAULT_MODEL = "gpt-4o-mini"

# Conversation states
TOPIC_INPUT = 1

# Function to initialize database and tables
async def init_db():
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

        await conn.commit()

# Function to migrate database (add 'deleted' column if not exists)
async def migrate_db():
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()

        # Check if 'deleted' column exists in 'topics' table
        await c.execute("PRAGMA table_info(topics)")
        columns = await c.fetchall()
        column_names = [column[1] for column in columns]
        if 'deleted' not in column_names:
            await c.execute("ALTER TABLE topics ADD COLUMN deleted INTEGER DEFAULT 0")
            await conn.commit()

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
    return len(ENCODER.encode(text))

# Function to get conversation history
async def get_conversation_history(user_id, topic, max_tokens):
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
    """Set the model for the user."""
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
    final_prompt = prompt + "\n Respond with Markdown. No more than 4000 characters."
    # Ensure conversation_history is not None
    if not conversation_history:
        conversation_history = []

    provider = MODELS_INFO[selected_model]["provider"]
    api_model = MODELS_INFO[selected_model]["api_model"]

    try:
        if provider == "openai":
            # OpenAI's response generation remains unchanged
            full_conversation = conversation_history + [{"role": "user", "content": final_prompt}]
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=api_model,
                messages=full_conversation
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "gemini":
            # Using Gemini's API for response generation
            model = genai.GenerativeModel(api_model)
            # Need to format the conversation history for Gemini "content" to "parts", "assistant" to "model"
            conversation_history = [{"parts": [msg["content"]], "role": "model" if msg["role"] == "assistant" else "user"} for msg in conversation_history]
            chat = model.start_chat(history=conversation_history)
            response = chat.send_message(final_prompt)
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
# Reformat text for MarkdownV2
def reformat_markdown_to_v2(text: str) -> str:
    def escape(text, exclude_formatting=False):
        special_chars = '_*[]()~`>#+-=|{}.!'
        if exclude_formatting:
            special_chars = special_chars.replace('_', '').replace('*', '').replace('~', '').replace('`', '').replace('>', '')
        escaped_text = ''
        for c in text:
            if c == '\\':
                escaped_text += '\\\\'
            elif c in special_chars:
                escaped_text += '\\' + c
            else:
                escaped_text += c
        return escaped_text

    def extract_code_blocks(text):
        code_blocks = []

        code_block_pattern = re.compile(r'```.*?```', re.DOTALL)

        def replace_code_block(match):
            code_content = match.group(0)
            placeholder = f"{{CODEBLOCK{len(code_blocks)}}}"
            code_blocks.append(code_content)
            return placeholder

        text = code_block_pattern.sub(replace_code_block, text)
        return text, code_blocks

    def extract_code_spans(text):
        code_spans = []

        code_span_pattern = re.compile(r'`[^`]*`')

        def replace_code_span(match):
            code_content = match.group(0)
            placeholder = f"{{CODESPAN{len(code_spans)}}}"
            code_spans.append(code_content)
            return placeholder

        text = code_span_pattern.sub(replace_code_span, text)
        return text, code_spans

    def process_headers(text):
        header_pattern = re.compile(r'^(#{1,6})\s*(.*)', re.MULTILINE)

        def replace_header(match):
            header_text = match.group(2)
            escaped_text = escape(header_text)
            formatted_text = '_*' + escaped_text + '*_'
            return formatted_text

        text = header_pattern.sub(replace_header, text)
        return text

    def process_bold_italic(text):
        bold_italic_pattern = re.compile(r'(\*\*\*|___)(.+?)\1')

        def replace_bold_italic(match):
            content = match.group(2)
            escaped_content = escape(content)
            formatted_content = '_*' + escaped_content + '*_'
            return formatted_content

        text = bold_italic_pattern.sub(replace_bold_italic, text)
        return text

    def process_bold(text):
        bold_pattern = re.compile(r'(\*\*|__)(.+?)\1')

        def replace_bold(match):
            content = match.group(2)
            escaped_content = escape(content)
            formatted_content = '*' + escaped_content + '*'
            return formatted_content

        text = bold_pattern.sub(replace_bold, text)
        return text

    def process_italic(text):
        italic_pattern = re.compile(r'(\*|_)(.+?)\1')

        def replace_italic(match):
            content = match.group(2)
            escaped_content = escape(content)
            formatted_content = '_' + escaped_content + '_'
            return formatted_content

        text = italic_pattern.sub(replace_italic, text)
        return text

    def process_strikethrough(text):
        strikethrough_pattern = re.compile(r'~~(.+?)~~')

        def replace_strikethrough(match):
            content = match.group(1)
            escaped_content = escape(content)
            formatted_content = '~' + escaped_content + '~'
            return formatted_content

        text = strikethrough_pattern.sub(replace_strikethrough, text)
        return text

    def process_links(text):
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        def replace_link(match):
            link_text = match.group(1)
            link_url = match.group(2)
            escaped_text = escape(link_text)
            link_url = link_url.replace('\\', '\\\\').replace(')', '\\)')
            formatted_link = '[' + escaped_text + '](' + link_url + ')'
            return formatted_link

        text = link_pattern.sub(replace_link, text)
        return text

    def process_images(text):
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

        def replace_image(match):
            alt_text = match.group(1)
            escaped_text = escape(alt_text)
            return escaped_text

        text = image_pattern.sub(replace_image, text)
        return text

    def process_block_quotes(text):
        block_quote_pattern = re.compile(r'^(>+)\s*(.*)', re.MULTILINE)

        def replace_block_quote(match):
            content = match.group(2)
            escaped_content = escape(content)
            formatted_content = match.group(1) + ' ' + escaped_content
            return formatted_content

        text = block_quote_pattern.sub(replace_block_quote, text)
        return text

    def process_code_blocks(code_blocks):
        processed_blocks = []

        for block in code_blocks:
            code_content = block.strip('`').strip()
            code_content = code_content.replace('\\', '\\\\').replace('`', '\\`')
            formatted_block = '```' + code_content + '```'
            processed_blocks.append(formatted_block)

        return processed_blocks

    def process_code_spans(code_spans):
        processed_spans = []

        for span in code_spans:
            code_content = span.strip('`')
            code_content = code_content.replace('\\', '\\\\').replace('`', '\\`')
            formatted_span = '`' + code_content + '`'
            processed_spans.append(formatted_span)

        return processed_spans

    # Extract and process code blocks and spans
    text, code_blocks = extract_code_blocks(text)
    text, code_spans = extract_code_spans(text)

    # Process other markdown elements
    text = process_images(text)
    text = process_links(text)
    text = process_strikethrough(text)
    text = process_bold_italic(text)
    text = process_bold(text)
    text = process_italic(text)
    text = process_headers(text)
    text = process_block_quotes(text)

    # Escape any remaining special characters
    text = escape(text, exclude_formatting=True)

    # Process code blocks and spans
    processed_code_blocks = process_code_blocks(code_blocks)
    processed_code_spans = process_code_spans(code_spans)

    # Replace placeholders with processed code blocks and spans
    for i, code_block in enumerate(processed_code_blocks):
        placeholder = f"{{CODEBLOCK{i}}}"
        text = text.replace(placeholder, code_block)

    for i, code_span in enumerate(processed_code_spans):
        placeholder = f"{{CODESPAN{i}}}"
        text = text.replace(placeholder, code_span)

    return text


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
    markdown_response = reformat_markdown_to_v2(ai_response)
    await update.message.reply_text(markdown_response, parse_mode="MarkdownV2")

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
async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to the bot!")

async def start_new(update: Update, context) -> int:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)

    await update.message.reply_text("Please enter a topic name for the new chat, or type 'auto' to generate one automatically.")
    return TOPIC_INPUT

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

async def help_handle(update: Update, context) -> None:
    user_id = update.message.from_user.id
    current_model = await get_user_model(user_id)
    current_topic = await get_current_topic(user_id)
    await update.message.reply_text(
        f"**Current Model:** {MODELS_INFO[current_model]['name']}"
        f"\n\n**Topic:** {current_topic}"
        "\n\n**Commands:**"
        "\n/new - Start new dialog"
        "\n/settings - Show settings"
        "\n/topic - Show current topic"
        "\n/help - Show help message",
        parse_mode="Markdown"
    )

async def show_topic(update: Update, context) -> None:
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)
    topic = await get_current_topic(user_id)
    await update.message.reply_text(f"**Current topic:** {topic}", parse_mode="Markdown")

async def post_init(application):
    await init_db()
    await migrate_db()  # Ensure migration is applied
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/topic", "Show current topic"),
        BotCommand("/help", "Show help message"),
    ])

async def get_settings_menu_async(user_id):
    current_model = await get_user_model(user_id)
    current_topic = await get_current_topic(user_id)

    text = f"**Current Model:** {MODELS_INFO[current_model]['name']}\n"
    text += f"**Current Topic:** {current_topic}\n\n"

    text += f"{MODELS_INFO[current_model]['description']}\n\n"

    text += "**Model Scores:**\n"
    score_dict = MODELS_INFO[current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += f"{score_key.capitalize()} : {score_value}\n"
    text += "\nCost per 1M tokens:\n"
    text += f"Input: ${MODELS_INFO[current_model]['cost_input']*COST_INTEREST_RATE}\n"
    text += f"Output: ${MODELS_INFO[current_model]['cost_output']*COST_INTEREST_RATE}\n\n"
    text += "**Select Model:**"

    # Buttons for models
    model_buttons = []
    for model_key in MODELS_INFO:
        title = MODELS_INFO[model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        model_buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_model|{model_key}")
        )

    # Fetch active topics
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        c = await conn.cursor()
        await c.execute("SELECT topic FROM topics WHERE user_id=? AND deleted=0", (user_id,))
        topics = await c.fetchall()

    topic_buttons = []
    for (topic_name,) in topics:
        if topic_name == current_topic:
            title = "✅ " + topic_name
        else:
            title = topic_name
        topic_buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_topic|{topic_name}")
        )

    # Add a button to delete topics
    delete_topic_button = InlineKeyboardButton("🗑 Delete Topic", callback_data="delete_topic_menu")

    # Organize buttons
    reply_markup = InlineKeyboardMarkup([
        model_buttons,
        *[topic_buttons[i:i+2] for i in range(0, len(topic_buttons), 2)],
        [delete_topic_button]
    ])

    return text, reply_markup

async def settings_handle(update: Update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    await register_user(user_id, username)
    
    text, reply_markup = await get_settings_menu_async(user_id)

    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

async def set_model_handle(update: Update, context):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Get selected model from callback data
    _, model_key = query.data.split("|")

    # Set the user's selected model in the database
    await set_user_model(user_id, model_key)

    # Get updated settings menu
    text, reply_markup = await get_settings_menu_async(user_id)

    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    except error.BadRequest as e:
        if "Message is not modified" not in str(e):
            print(f"Error editing message: {e}")

async def set_topic_handle(update: Update, context):
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

        # Get updated settings menu
        text, reply_markup = await get_settings_menu_async(user_id)

        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
        except error.BadRequest as e:
            if "Message is not modified" not in str(e):
                print(f"Error editing message: {e}")
    else:
        await query.answer("Topic not found or has been deleted.")

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
            InlineKeyboardButton("🗑 " + topic_name, callback_data=f"delete_topic|{topic_name}")
        )

    # Organize buttons in rows of 2
    button_rows = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    reply_markup = InlineKeyboardMarkup(button_rows)

    await query.edit_message_text("Select a topic to delete:", reply_markup=reply_markup)

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

    # Get updated settings menu
    text, reply_markup = await get_settings_menu_async(user_id)

    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    await query.answer("Topic deleted.")

### MAIN ###
if __name__ == '__main__':
    # Create the bot application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Conversation handler for /new command
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('new', start_new)],
        states={
            TOPIC_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_topic_name)],
        },
        fallbacks=[],
    )

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_handle))
    app.add_handler(CommandHandler("topic", show_topic))
    app.add_handler(CommandHandler("settings", settings_handle))
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(set_model_handle, pattern="^set_model\|"))
    app.add_handler(CallbackQueryHandler(set_topic_handle, pattern="^set_topic\|"))
    app.add_handler(CallbackQueryHandler(delete_topic_menu_handle, pattern="^delete_topic_menu"))
    app.add_handler(CallbackQueryHandler(delete_topic_handle, pattern="^delete_topic\|"))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    # Start the bot (this will run the event loop)
    app.run_polling()
