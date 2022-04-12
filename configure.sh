#!/bin/bash

# Configuring virtual environment
if [ ! -d venv ]; then
    echo "Configuring virtual environment"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python3 -m spacy download pt_core_news_lg
    deactivate
    echo "Virtual environment configured"
fi