# PlantCure - Multilingual AI Plant Disease Detection

PlantCure is a full-stack, multilingual final-year project that detects plant leaf diseases from uploaded/captured images, stores user-wise diagnosis history, shows analytics, and provides AI assistant guidance for treatment and next actions. It features real-time language localization for Hindi, Marathi, and English.

## Key Features

- **Multilingual Support**: Switch seamlessly between English, Hindi, and Marathi.
- **Single Page Application (SPA)**: Fast, dynamic navigation without page reloads.
- **User Authentication**: Register/login/logout with role support.
- **Leaf Diagnosis**: Supports both file drop/upload and live web-camera capture.
- **Quality Control**: Invalid image rejection for non-leaf or low-confidence images.
- **Dashboards & Analytics**: KPI panels and donut charts tracking your plant health over time.
- **Multilingual AI Assistant**: Chatbot for treatment and fertilizer guidance that responds in your chosen language using the Gemini API.
- **Custom Training**: Training script for dataset-based model creation with fine-tuning.

## Tech Stack

- **Backend**: Python 3.11+, Flask 3.1+, SQLAlchemy, Flask-Login
- **Frontend**: HTML5, Vanilla JavaScript, Vanilla CSS Variables
- **AI/ML**: TensorFlow/Keras 2.15, Pillow, google-generativeai, googletrans
- **Database**: SQLite

## Project Structure

```text
plantcure/
  app.py
  model_loader.py
  chatbot.py
  disease_knowledge.py
  requirements.txt
  .env.example
  lang/                 <-- JSON files for UI localization (en, hi, mr)
  templates/
    main_spa.html       <-- Centralized Single Page Application UI
    auth/
  static/
  models/
  uploads/
  instance/
```

## Quick Start (Windows)

1. Create and activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Setup environment variables
Create a `.env` file in the root directory (you can copy `.env.example`):
```text
SECRET_KEY=your_random_secret_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
*Note: The Gemini API key is free of cost up to 15 requests/minute. Get it from Google AI Studio.*

4. Run the application
```powershell
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

## Run With Existing Model

If `models/combined_plant_disease_model` exists, the app loads it automatically.
It also loads `models/class_labels.json` for class ordering.

## Core APIs

- `POST /api/predict` - Handles image diagnosis and automatic translation to the chosen language.
- `GET /api/lang/<code_id>` - Supplies JSON dictionary for dynamic UI localization (en, hi, mr).
- `POST /api/chat` - Translates user query to English, processes it via the Gemini API PlantChatbot, and translates the response back.
- `GET /api/history` - Recent diagnosis records for the logged-in user.
- `GET /api/analytics` - Aggregate statistics for the analytics dashboard charts.

## Common Issues

- `ModuleNotFoundError`: Run `pip install -r requirements.txt` again. Ensure `.venv` is active.
- **Camera Black Feed**: Check your browser and OS permissions for camera access.
- **Translation Errors**: If you encounter a `googletrans` error, make sure you are using `googletrans==4.0.0-rc1` as specified in requirements.
- **503 Model Unavailable**: The system may not be able to load the `.keras` or `.h5` file. Verify the file isn't corrupted and matches the TensorFlow version.

## Supporting Docs

- Comprehensive Setup & User Guide: `GUIDE.md`
- Metrics template: `docs/RESULT_METRICS_TEMPLATE.md`
- Viva Q&A sheet: `docs/VIVA_QA_SHEET.md`
