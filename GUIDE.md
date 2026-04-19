# 🌿 PlantCure: User Guide & Implementation Walkthrough

Welcome to **PlantCure v3**. This guide covers how to get the most accurate results from the AI and provides a technical breakdown of the system implementation.

---

## 📖 User Guide: How to Get Accurate Results

To ensure the AI identifies your plant diseases correctly, follow these specific steps:

### 📸 1. Selecting the Right Image
The model is trained on the **PlantVillage** dataset. For the highest accuracy:
*   **Recommended Source**: Navigate to your local `plantvillage` dataset folder.
*   **Step**: Select a crop folder (e.g., `Tomato___Early_blight`) and choose any `.jpg` image from it.
*   **Why?**: These images are standardized, which the model recognizes perfectly.

### 🔍 2. Using External (Google) Images
If you are using images found on the internet:
*   **Background**: Ensure the background is **clear and solid** (preferably black, white, or neutral). Busy backgrounds (other plants, hands, or dirt) can confuse the model.
*   **Quality**: The image must be clear, sharp, and focused on a **single leaf**.
*   **Lighting**: Ensure the leaf is well-lit without harsh shadows or glares.

### 🚀 3. Analysis Process
1.  **Browse**: Click "Browse file" or drag an image into the drop zone.
2.  **Verify**: Look at the large preview. If the leaf is blurry, click **Try Again**.
3.  **Analyse**: Click **Analyse Plant**.
4.  **Review**: If confidence is low (below 70%), check the **Low Confidence Warning** box for specific issues and retry with a better photo.

---

## 🛠️ Step-by-Step Implementation Guide

The following steps were taken to build and stabilize the current version of PlantCure:

### Phase 1: Model Stabilization
1.  **H5 Model Patching**: Created `scripts/fix_model.py` to translate newer Keras 3 configurations (like `batch_shape`) into the older `batch_input_shape` format compatible with TensorFlow 2.15.
2.  **Label Synchronization**: Synchronized `class_labels.json` with the model's actual output layers and mapped them to the `KAGGLE_MAP` in `model_loader.py`.

### Phase 2: Enhanced Prediction Engine
1.  **Test-Time Augmentation (TTA)**: Implemented in `model_loader.py`. The system now creates 3 variants of your uploaded image (flipped, rotated, brightened) and averages the predictions for higher stability.
2.  **Bias Correction Layer**: Added logic to detect "Pepper" bias. If the model incorrectly predicts Pepper but sees a secondary match for the actual crop (e.g., Apple), it automatically pivots to the correct result.
3.  **Confidence Gate**: Implemented a 70% threshold. Predictions below this are flagged as "Uncertain" to prevent providing incorrect treatment advice.

### Phase 3: Single Page Architecture (SPA)
1.  **Unified Routing**: Refactored `app.py` to route `/dashboard`, `/detect`, `/history`, and `/analytics` to a single `main_spa.html` file.
2.  **View Management**: Implemented a JavaScript `switchView` system to swap content blocks without page reloads, fixing all previous 404 errors.

### Phase 4: UI/UX & Camera Integration
1.  **Native Camera**: Integrated the MediaDevices API to allow in-browser photo capture with a dedicated fullscreen overlay.
2.  **Contextual Chat**: 
    -   Integrated `DiseaseKnowledgeBase` into the Chatbot.
    -   Implemented "Chat Memory" so the assistant knows about the last scanned result even if the user asks a general question like "How do I treat it?".
3.  **Uncertainty UI**: Added specialized warning blocks that display "Issues Detected" and "Suggestions to improve" based on real-time AI feedback.

### Phase 5: Knowledge Base & Chatbot
1.  **Rich KB Integration**: Mapped the detection results to a localized `knowledge_base.json` containing symptoms, prevention, organic remedies, and chemical treatments.
2.  **API Fallback**: Configured the chatbot to use local rules if the Gemini AI API is unavailable, ensuring the user always gets a response.

---

**Note**: To run the application, ensure your virtual environment is active and run `python app.py`. Access the dashboard at `http://localhost:5000`.

**Note**: To add GEMINI_API_KEY its free of cost upto 15 request.

**Note for .env file**: 
1. Create a .env file in the root directory of the project.
2. Add the following lines to the .env file:
   GEMINI_API_KEY=your_gemini_api_key_here
   FLASK_SECRET_KEY=your_random_secret_key_here
3. Replace the placeholder values with your actual values.
4. Run the application using `python app.py`.
**Setup for .env file**:
1. Set Flask SECRET_KEY:
1: Create .env file in your project root
SECRET_KEY=your_random_secret_key_here
GEMINI_API_KEY=your_gemini_api_key_here
2: Install dotenv
pip install python-dotenv
3: Load in Flask (app.py)
from dotenv import load_dotenv
import os
load_dotenv()

app.secret_key = os.getenv("SECRET_KEY")
2. Set Gemini API Key
1: Get API key from Google AI Studio
2: Load it in your code
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
3: Use it (example)
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
