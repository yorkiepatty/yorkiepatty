#!/bin/bash
# Setup Ollama for Sunny's Local AI Learning

echo "🦙 Setting up Ollama for Sunny's Autonomous Learning..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🚀 Starting Ollama service..."
ollama serve &

# Wait for service to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Pull recommended models for Sunny
echo "📚 Pulling AI models for Sunny..."

# Llama 3.2 - Good for general knowledge and reasoning
ollama pull llama3.2:latest
echo "✅ Llama 3.2 ready"

# Mistral - Good for coding and technical topics  
ollama pull mistral:latest
echo "✅ Mistral ready"

# Qwen - Good for math and science
ollama pull qwen:latest
echo "✅ Qwen ready"

echo ""
echo "🎉 Sunny's Local AI Learning Environment Ready!"
echo ""
echo "Available models:"
ollama list
echo ""
echo "🧠 Sunny can now learn autonomously using local AI models"
echo "   No internet or API keys required!"
echo ""
echo "To test: python3 autonomous_learning_engine.py"