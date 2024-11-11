#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess documents for different RAG systems
echo "Preprocessing documents for Chroma Azure RAG..."
python utils/preprocess.py --rag_type chroma --delete yes --collection_name chroma_docs

echo "Preprocessing documents for Azure Search RAG..."
python utils/preprocess.py --rag_type azure_search --delete yes --collection_name docs-index

echo "Preprocessing documents for Light RAG..."
python utils/preprocess.py --rag_type light --delete yes --collection_name light_docs

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py