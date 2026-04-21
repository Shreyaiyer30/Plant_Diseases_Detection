import os
import sys
import io

# Fix for Windows terminal UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# setup dummy env vars
os.environ["MONGODB_URI"] = "mongodb://localhost:27017"

from ai_chatbot import get_bot_reply

questions = [
    ("Tell me about tomato plants", "Tomato Late Blight"),
    ("Why is my plant yellow?", "Unknown")
]

for q, disease in questions:
    print(f"\n--- Question: {q} (Disease Context: {disease}) ---")
    try:
        reply = get_bot_reply(q, disease=disease)
        print("Reply:", reply)
    except Exception as e:
        print("Error:", e)
