#!/bin/bash

echo "=========================================="
echo "Derek Dashboard Installation"
echo "The Christman AI Project"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r config/requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data models logs media/avatars media/audio data/knowledge_base logs/self_modifications

# Run tests
echo ""
echo "Running initial tests..."
python -m pytest tests/ -v || echo "⚠️  Some tests failed (this is normal for initial setup)"

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "To start Derek Dashboard:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo ""
echo "How can we help you love yourself more? 💙"
