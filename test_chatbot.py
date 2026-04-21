import os
import sys
import io

# Fix for Windows terminal UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# setup dummy env vars
os.environ["MONGODB_URI"] = "mongodb://localhost:27017"

from ai_chatbot import get_bot_reply

questions = [
    ("How to treat early blight?", "Early Blight"),
    ("Organic remedy for powdery mildew?", "Powdery Mildew"),
    ("Prevention tips for rust?", "Rust")
]

for q, disease in questions:
    print(f"\n--- Question: {q} (Disease Context: {disease}) ---")
    try:
        reply = get_bot_reply(q, disease=disease)
        print("Reply:", reply)
    except Exception as e:
        print("Error:", e)

