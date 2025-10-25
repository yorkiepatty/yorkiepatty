import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("PERPLEXITY_API_KEY")
if api_key:
    # Only show first 10 chars for security
    print(f"API key found: {api_key[:10]}...")
else:
    print("No API key found!")

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
