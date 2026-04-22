# test_groq.py
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
print(f"API Key found: {bool(api_key)}")

try:
    from groq import Groq
    
    if api_key:
        client = Groq(api_key=api_key)
        
        # Use the NEW model name
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from Groq!'"}
            ],
            model="llama-3.3-70b-versatile",  # ← UPDATED MODEL
            temperature=0.5,
            max_tokens=50,
        )
        
        print("\n✅ SUCCESS! Groq response:")
        print(response.choices[0].message.content)
        
    else:
        print("❌ No API key found")
        
except Exception as e:
    print(f"❌ Error: {e}")