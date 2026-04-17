#!/usr/bin/env bash
set -e
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RESET="\033[0m"
echo -e "\n${GREEN}============================================${RESET}"
echo -e "${GREEN}  PlantCure v2 - Plant Disease Detection${RESET}"
echo -e "${GREEN}============================================${RESET}\n"
[ ! -d ".venv" ] && { echo -e "${YELLOW}[SETUP]${RESET} Creating virtual environment..."; python3 -m venv .venv; }
source .venv/bin/activate
echo -e "${YELLOW}[SETUP]${RESET} Installing dependencies..."
pip install -r requirements.txt -q
[ -f "models/plant_model.h5" ] && echo -e "${GREEN}[INFO]${RESET}  Trained model found." || echo -e "${YELLOW}[INFO]${RESET}  No model — Demo Mode."
echo -e "\n${GREEN}[START]${RESET} Open http://localhost:5000 in your browser."
echo -e "        Press Ctrl+C to stop.\n"
python3 app.py
