import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_chatbot import PlantChatbot


def run_demo():
    bot = PlantChatbot()
    session_id = "demo-session"
    test_cases = [
        ("How to treat this disease?", "Rust"),
        ("Organic remedies please", "Powdery Mildew"),
        ("How to prevent this from coming back?", "Blight"),
        ("What are the symptoms I should look for?", "Unknown"),
        ("Monsoon me kya care karni chahiye?", "Tomato Early Blight"),
    ]

    for question, disease in test_cases:
        response, source, meta = bot.get_response(
            user_message=question,
            disease=disease,
            session_id=session_id,
            confidence_score=0.945,
        )
        print("=" * 80)
        print(f"Disease : {disease}")
        print(f"Question: {question}")
        print(f"Source  : {source}")
        print(f"Intent  : {meta.get('intent')}")
        print(f"Reply   : {response}")


if __name__ == "__main__":
    run_demo()
