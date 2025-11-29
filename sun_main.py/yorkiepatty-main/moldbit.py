# Avatar image loader for Moldbit interface
import os
from pathlib import Path

def load_image(filename: str):
    """Load an image file from the assets directory"""
    assets_dir = Path(__file__).parent / "assets" / "images"
    image_path = assets_dir / filename
    
    if image_path.exists():
        return str(image_path)
    return None

talking_img = load_image("avatar_talking.png")
idle_img = load_image("avatar_idle.png")

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
