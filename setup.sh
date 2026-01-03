#!/usr/bin/env bash



# VS Code settings
mkdir -p .vscode
touch .vscode/settings.json

# GitHub workflows
mkdir -p .github/workflows
touch .github/workflows/unittests.yml

# Data directories
mkdir -p data/raw
mkdir -p data/processed

# Vector store (FAISS / ChromaDB)
mkdir -p vector_store

# Notebooks
mkdir -p notebooks
touch notebooks/__init__.py
touch notebooks/README.md

# Source code
mkdir -p src
touch src/__init__.py

# Tests
mkdir -p tests
touch tests/__init__.py

# App and configs
touch app.py
touch requirements.txt
touch README.md
touch .gitignore

echo "Project structure '$PROJECT_NAME' created successfully."
