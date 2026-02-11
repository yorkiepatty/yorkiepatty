"""
Royce AI Image Generator
------------------------
Generates images from text descriptions using AI.
Supports DALL-E (OpenAI) and Stable Diffusion (via Replicate).
"""

import os
import json
import re
import logging
import base64
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RoyceImages:
    """AI image generation from text descriptions."""

    def __init__(self):
        from .config import RoyceConfig
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.replicate_key = os.getenv("REPLICATE_API_KEY")
        self.images_dir = Path(str(RoyceConfig.IMAGES_DIR))
        self.images_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Royce Image Generator initialized")

    def generate(self, description: str, style: str = "vivid", size: str = "1024x1024") -> Dict:
        """Generate an image from a text description.

        Args:
            description: What to create
            style: "vivid" or "natural"
            size: "1024x1024", "1792x1024", or "1024x1792"

        Returns:
            Dict with image path and metadata
        """
        if self.openai_key:
            return self._generate_dalle(description, style, size)
        elif self.replicate_key:
            return self._generate_replicate(description)
        else:
            return {
                "status": "error",
                "message": "I need either an OpenAI API key (for DALL-E) or a Replicate API key to generate images. Add one to your .env file."
            }

    def _generate_dalle(self, description: str, style: str = "vivid", size: str = "1024x1024") -> Dict:
        """Generate using OpenAI DALL-E 3."""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "dall-e-3",
                "prompt": description,
                "n": 1,
                "size": size,
                "style": style,
                "response_format": "url",
            }

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            image_url = result["data"][0]["url"]
            revised_prompt = result["data"][0].get("revised_prompt", description)

            # Download and save locally
            image_response = requests.get(image_url, timeout=30)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"royce_image_{timestamp}.png"
            filepath = self.images_dir / filename

            with open(filepath, "wb") as f:
                f.write(image_response.content)

            return {
                "status": "success",
                "description": description,
                "revised_prompt": revised_prompt,
                "file_path": str(filepath),
                "url": image_url,
                "model": "dall-e-3",
                "message": f"Created the image and saved it to {filepath}",
            }

        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if "content_policy" in error_msg.lower():
                return {"status": "error", "message": "That description got flagged by the content policy. Try describing it differently."}
            logger.error(f"DALL-E error: {e}")
            return {"status": "error", "message": f"Image generation failed: {error_msg}"}
        except Exception as e:
            logger.error(f"DALL-E error: {e}")
            return {"status": "error", "message": f"Something went wrong generating that image: {str(e)}"}

    def _generate_replicate(self, description: str) -> Dict:
        """Generate using Stable Diffusion via Replicate."""
        try:
            headers = {
                "Authorization": f"Token {self.replicate_key}",
                "Content-Type": "application/json",
            }
            data = {
                "version": "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
                "input": {
                    "prompt": description,
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                }
            }

            # Start prediction
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=data,
                timeout=10,
            )
            response.raise_for_status()
            prediction = response.json()
            prediction_url = prediction["urls"]["get"]

            # Poll for result
            import time
            for _ in range(60):
                time.sleep(2)
                poll = requests.get(prediction_url, headers=headers, timeout=10)
                poll_data = poll.json()

                if poll_data["status"] == "succeeded":
                    image_url = poll_data["output"][0]

                    # Download and save
                    image_response = requests.get(image_url, timeout=30)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"royce_image_{timestamp}.png"
                    filepath = self.images_dir / filename

                    with open(filepath, "wb") as f:
                        f.write(image_response.content)

                    return {
                        "status": "success",
                        "description": description,
                        "file_path": str(filepath),
                        "url": image_url,
                        "model": "stable-diffusion",
                        "message": f"Created the image and saved it to {filepath}",
                    }
                elif poll_data["status"] == "failed":
                    return {"status": "error", "message": "Image generation failed on the server side."}

            return {"status": "error", "message": "Image generation timed out."}

        except Exception as e:
            logger.error(f"Replicate error: {e}")
            return {"status": "error", "message": f"Image generation failed: {str(e)}"}

    def parse_image_command(self, text: str) -> Optional[Dict]:
        """Detect image generation requests from natural language."""
        text_lower = text.lower().strip()

        patterns = [
            r"(?:make|create|generate|draw|paint|design) (?:me |an? )?(?:image|picture|photo|art|illustration) (?:of |showing |with |that shows )?(.+)",
            r"(?:make|create|generate|draw|paint) (.+)",
            r"(?:can you|could you) (?:make|create|generate|draw) (.+)",
            r"i want (?:an? )?(?:image|picture|photo) (?:of |showing )?(.+)",
            r"picture (?:of |showing )(.+)",
        ]

        for pattern in patterns:
            match = re.match(pattern, text_lower)
            if match:
                desc = match.group(1).strip().rstrip("?.!")
                if len(desc) > 5:  # Avoid triggering on very short matches
                    return {"action": "generate", "description": desc}

        return None
