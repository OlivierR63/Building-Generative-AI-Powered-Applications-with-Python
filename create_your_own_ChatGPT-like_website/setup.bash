#!/bin/bash
source ../setup.bash

echo "\nInstall flask\n"
python3 -m pip install flask

echo "\nInstall flask_cors\n"
python3 -m pip install flask_cors

echo "\nInstall transformers and torch\n"
python3 -m pip install transformers torch

echo "\n Install packages enumerated in LLM_application_chatbot/requirements.txt\n"
python3 -m pip install -r LLM_application_chatbot/requirements.txt