#!/bin/bash
# Derek Ultimate Voice - Quick Start Script
# Run this after setting up your .env file

echo "🚀 Derek Ultimate Voice - Quick Start"
echo "====================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "📋 Creating .env from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - ANTHROPIC_API_KEY"
    echo "   - OPENAI_API_KEY"
    echo "   - PERPLEXITY_API_KEY"
    echo "   - AWS_ACCESS_KEY_ID"
    echo "   - AWS_SECRET_ACCESS_KEY"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if API keys are set
source .env
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo "⚠️  API keys not configured in .env file"
    echo "Please edit .env and add your actual API keys"
    exit 1
fi

echo "✅ .env file found and configured"
echo ""

# Install dependencies
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Test imports
echo "🔍 Testing imports..."
python3 -c "
import derek_ultimate_voice
print('✅ All systems operational!')
" 2>&1 | grep -E '✅|❌|⚠️'

echo ""
echo "🎤 Derek Ultimate Voice is ready!"
echo ""
echo "To run Derek:"
echo "  python3 derek_ultimate_voice.py"
echo ""
echo "Or import in your code:"
echo "  from derek_ultimate_voice import DerekUltimateVoice"
echo "  derek = DerekUltimateVoice()"
echo ""

