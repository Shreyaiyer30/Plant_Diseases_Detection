# AI Chatbot Integration

## What was added

- `ai_chatbot.py`: context-aware chatbot class with Gemini integration, local knowledge-base fallback, session memory, caching, rate limiting, and optional MongoDB logging.
- `/api/chat`: now accepts `session_id` and `confidence`, uses the new chatbot, and stores the response in `ChatMessage`.
- `scripts/test_chatbot.py`: simple local test runner.

## Environment variables

Add these to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=plantcure
MONGODB_CHAT_COLLECTION=chatbot_interactions
```

MongoDB is optional. If `MONGODB_URI` is missing, chatbot logging continues only in the existing SQL table.

## Flask endpoint

### Request

`POST /api/chat`

```json
{
  "question": "How to treat this disease?",
  "disease": "Rust",
  "session_id": "user-42-rust-chat",
  "confidence": 0.945,
  "lang": "en"
}
```

### Response

```json
{
  "response": "🌿 Treatment for Rust: remove infected leaves, apply neem oil weekly, and use sulfur-based fungicide. Start with sanitation first, then spray in the early morning or evening.",
  "source": "gemini",
  "meta": {
    "detected_disease": "Rust",
    "intent": "treatment",
    "session_id": "user-42-rust-chat",
    "confidence_score": 0.945,
    "cached": false,
    "timestamp": "2026-04-20T00:00:00+00:00"
  },
  "session_id": "user-42-rust-chat"
}
```

## Frontend notes

- Keep sending the same `session_id` for follow-up questions so the chatbot can use recent context.
- Pass the CNN prediction confidence into `confidence`.
- Pass `lang` when the user is chatting in Hindi or another supported UI language.

## Testing

Run:

```powershell
python scripts/test_chatbot.py
```

If Gemini is not configured or fails, the chatbot falls back to the local knowledge base automatically.

## Maintenance

- Update `models/knowledge_base.json` to improve disease answers without changing code.
- If you want stronger crop-specific advice, add disease entries that match your model output names exactly.
- The rate limit is `60` requests per minute per session and the cache TTL is `1` hour; both can be adjusted in `PlantChatbot(...)`.
