# Sunny Voice Assistant

A sophisticated AI voice assistant powered by AWS Bedrock (Claude 3.5 Sonnet) with comprehensive file processing, vision capabilities, and action execution.

## Files in this repository:

- `assistant.py` - Main voice assistant application (1,636 lines)
- `.env` - Configuration file for API keys
- `README.md` - This file

## Features:
- 🤖 AI Chat with Claude 3.5 Sonnet
- 🎤 Voice input/output with AWS Polly
- 👁️ Vision (image/video analysis)
- 📄 File reading (PDF, Word, Excel, etc.)
- 🌐 Web reading and search
- 🎨 AI image/video generation
- 📧 Email integration
- 💾 Conversation memory

## Setup:
1. Install dependencies: `pip install -r requirements.txt`
2. Configure AWS credentials: `aws configure`
3. Run: `python assistant.py`