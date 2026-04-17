# PlantCure - AI Plant Disease Detection

PlantCure is a full-stack final-year project that detects plant leaf diseases from uploaded/captured images, stores user-wise diagnosis history, shows analytics, and provides AI assistant guidance for treatment and next actions.

## Key Features

- User authentication (register/login/logout) with role support.
- Leaf diagnosis from file upload and live camera capture.
- Invalid image rejection for non-leaf/low-confidence images.
- Diagnosis history, dashboard KPIs, and analytics endpoints.
- AI assistant chat for treatment, fertilizer guidance, and 7-day action steps.
- Training script for dataset-based model creation with fine-tuning.

## Tech Stack

- Backend: Python, Flask, SQLAlchemy, Flask-Login
- Frontend: HTML, CSS, Vanilla JavaScript
- ML: TensorFlow/Keras, Pillow, NumPy, SciPy
- Database: SQLite

## Project Structure

```text
plantcure/
  app.py
  model_loader.py
  train_model.py
  requirements.txt
  templates/
  static/
  data/PlantVillage/
  models/
  uploads/
  instance/
```

## Quick Start (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

## Run With Existing Model

If `models/plant_model.h5` exists, the app loads it automatically.
It also loads `models/class_labels.json` for class ordering.

## Train Model on Dataset

### 1) Place dataset

Put dataset inside:

```text
data/PlantVillage/
```

Nested folders are handled (for example `data/PlantVillage/PlantVillage/...`).

### 2) Start training

```powershell
python train_model.py --dataset data/PlantVillage --epochs 30 --batch 16 --fine-tune --label-smoothing 0.08
```

Artifacts created:

- `models/plant_model.h5`
- `models/class_labels.json`
- `training_logs/*.csv`

### 3) Restart app

```powershell
python app.py
```

## Core APIs

- `POST /api/predict` - image diagnosis (returns invalid-image status for wrong uploads)
- `POST /api/chat` - disease-aware AI assistant response
- `GET /api/history` - recent diagnosis records
- `GET /api/analytics` - aggregate statistics
- `GET /api/health` - health check

## Notes for Final-Year Demo

- Camera capture works on secure context (`localhost` during local demo).
- Invalid uploads (hand/background/non-leaf) are rejected to avoid false disease output.
- Old missing thumbnails can occur if legacy uploaded files were deleted.
- Training on CPU is slow; allow long runtime for full epochs.

## Common Issues

- `ModuleNotFoundError`: run `pip install -r requirements.txt`
- Camera black feed: check browser/device permissions and switch camera in modal.
- Training only on partial classes: verify all class folders are present in dataset.

## Supporting Docs

- Metrics template: `RESULT_METRICS_TEMPLATE.md`
- Viva Q&A sheet: `VIVA_QA_SHEET.md`
