"""Step 1: Hello API - talk to Gemini from Python.

Run from the repo root:
    .venv\\Scripts\\python 01-hello-api\\hello_gemini.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load GEMINI_API_KEY (and GEMINI_MODEL) from the .env file into environment variables.
load_dotenv()

# "lite" models are smaller and cheaper, and usually less overloaded on the free tier.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# The client automatically reads GEMINI_API_KEY from the environment.
# Retry options: if Google answers 503 (overloaded) or 429 (rate limit), wait and try again
# up to 5 times, waiting longer each time (1s, 2s, 4s, ...) - called "exponential backoff".
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=1.0,
            http_status_codes=[429, 503],
        )
    )
)


def simple_question():
    """1. The most basic call: send text, get text back."""
    response = client.models.generate_content(
        model=MODEL,
        contents="Explain what an API is in one sentence, for a beginner.",
    )
    print("=== 1. Simple question ===")
    print(response.text)


def with_system_instruction():
    """2. A system instruction controls *how* the model behaves - your first 'tweak'."""
    response = client.models.generate_content(
        model=MODEL,
        contents="How do I reset my password?",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a friendly support agent for MicoProjects. "
                "Answer in at most 2 sentences. If you don't know something, say so."
            ),
            temperature=0.2,  # lower = more focused/consistent, higher = more creative
        ),
    )
    print("\n=== 2. With a system instruction ===")
    print(response.text)


def chat_with_memory():
    """3. A chat session keeps history, so the model remembers earlier messages."""
    chat = client.chats.create(model=MODEL)
    print("\n=== 3. Interactive chat (type 'exit' to quit) ===")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", ""}:
            break
        response = chat.send_message(user_input)
        print(f"Gemini: {response.text}\n")


if __name__ == "__main__":
    simple_question()
    with_system_instruction()
    chat_with_memory()
