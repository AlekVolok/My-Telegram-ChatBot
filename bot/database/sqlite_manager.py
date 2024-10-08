import sqlite3
import time
import tiktoken
import os

DATABASE_PATH = 'bot/database/chatbot.db'
encoding = tiktoken.get_encoding("cl100k_base")
ENCODER = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS = 100000  # Maximum allowed token count

# Function to initialize database and tables
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Create table to store users
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE
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


def register_user(username):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Check if user exists
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    result = c.fetchone()

    if result:
        user_id = result[0]  # Extract the id from the tuple
    else:
        # Insert user into the database
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = c.lastrowid  # Get the last inserted user ID

    conn.close()
    return user_id


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