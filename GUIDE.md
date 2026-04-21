Comprehensive Guide: Training, Testing, and Chatbot Setup
This guide provides the necessary commands and steps to manage the PlantCure AI model, install dependencies, and configure the multilingual chatbot.

📋 1. Requirements & Package Installation
To run the system, you need Python 3.11+. It is highly recommended to use a virtual environment to avoid package conflicts.

Initial Setup (One-time)
powershell
# Create a virtual environment
python -m venv .venv
# Activate the virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate
# Install essential dependencies
pip install -r requirements.txt
# Install additional AI & Database packages
pip install groq pymongo gdown googletrans==4.0.0rc1
Package Checklist
AI/ML: tensorflow==2.15.0, keras==2.15.0, opencv-python, numpy, gdown pipreqs
Backend: Flask, Flask-Login, flask-sqlalchemy, python-dotenv
Chatbot & Translation: 

groq
, googletrans, pymongo (for chat history)
🏋️ 2. Training the Model
If you have added new data or want to improve accuracy, you can retraining the CNN model.

Command:

powershell
python scripts/train_model.py --dataset data/PlantVillage --epochs 10 --batch 64 --img-size 128
--dataset: Path to your image folders (ensure data/PlantVillage exists).
--epochs: Number of training rounds (use 10–20 for a balance of speed and accuracy).
--img-size: Resolution (128x128 is optimized for standard hardware).
Result: The trained model will be saved to models/combined_plant_disease_model.
🧪 3. Testing the Model
You can verify the model's accuracy on individual images without starting the full web server.

Interactive Mode (Asks for path)
powershell
python scripts/test_model.py
Direct Command (Test a specific file)
powershell
python scripts/test_model.py --image path/to/leaf_image.jpg
🤖 4. Chatbot Configuration
The PlantCure chatbot is a context-aware assistant that provides treatment advice.

How it works:
Groq AI: Uses High-speed Llama-3 models for human-like conversation.
Local Knowledge Base: If API is down, it uses disease_knowledge.py to provide verified remedies.
Multilingual: Automatically detects if you are using Hindi or Marathi and translates responses.
Setting up the Chatbot API:
Go to Groq Console and get a free API Key.
Open your .env file and add:
text
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here  # Fallback
🚀 5. Running the Application
Once everything is installed and your .env is configured:

powershell
python app.py
Open http://localhost:5000 in your browser to start diagnosing plants.

🌐 6. Google Drive Model Access
The system can automatically download the model from Google Drive if it's missing.
1. Open `model_loader.py`.
2. Locate the `GD_MODEL_ID` variable at the top.
3. Replace the placeholder with your actual Google Drive File ID.
   * Example: If your link is `drive.google.com/file/d/1ABC-123/view`, the ID is `1ABC-123`.
4. The system will download the model to `models/combined_plant_model_final.h5` upon first run.

Work Summary
Analyzed the codebase to identify the exact training and testing scripts.
Verified the data/ directory structure for compatibility with training commands.
Extracted dependency requirements from requirements.txt and source code imports.
Documented the Chatbot logic, including its Groq AI integration and local fallback system.
Created a clear, step-by-step guide for installation, training, and testing.