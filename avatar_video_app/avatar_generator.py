"""
Avatar Generator - Creates avatars from text descriptions
Uses multiple AI image generation providers with fallback
"""
import os
import base64
import json
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib

from .config import config


@dataclass
class AvatarResult:
    """Result from avatar generation"""
    success: bool
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    provider: Optional[str] = None
    style: Optional[str] = None
    prompt: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AvatarGenerator:
    """
    Generates avatars from text descriptions using AI image generation.
    Supports multiple providers with intelligent fallback.
    """

    def __init__(self):
        self.config = config
        self.cache_dir = Path(config.temp_dir) / "avatar_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_providers(self) -> List[Dict[str, Any]]:
        """Initialize available image generation providers"""
        providers = []

        # Re-check the API key each time (in case it was loaded after init)
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")

        # OpenAI DALL-E 3
        if api_key:
            providers.append({
                "name": "openai_dalle3",
                "enabled": True,
                "priority": 1,
                "endpoint": "https://api.openai.com/v1/images/generations",
                "model": "dall-e-3",
                "api_key": api_key
            })

        # Add more providers as needed
        providers.append({
            "name": "placeholder",
            "enabled": True,
            "priority": 99,
            "description": "Generates placeholder avatars for testing"
        })

        return sorted(providers, key=lambda x: x.get("priority", 99))

    def _build_avatar_prompt(self, description: str, style: str = "realistic") -> str:
        """Build an optimized prompt for avatar generation"""
        style_prompts = {
            "realistic": "ultra realistic photograph, professional headshot, studio lighting, 8k resolution, detailed facial features",
            "anime": "anime style character portrait, vibrant colors, expressive eyes, clean linework, professional anime art",
            "cartoon": "cartoon character portrait, bold outlines, bright colors, friendly expression, Disney Pixar style",
            "3d_render": "3D rendered character portrait, Pixar style, high quality render, soft lighting, detailed textures",
            "artistic": "artistic portrait painting, impressionist style, beautiful brushwork, gallery quality",
            "pixel_art": "pixel art character portrait, 32-bit style, retro gaming aesthetic, clean pixels",
            "watercolor": "watercolor portrait painting, soft edges, beautiful color blending, artistic",
            "oil_painting": "oil painting portrait, classical style, rich colors, museum quality, renaissance inspired"
        }

        style_addition = style_prompts.get(style, style_prompts["realistic"])

        # Build comprehensive prompt
        prompt = f"""Create a portrait avatar of: {description}

Style: {style_addition}

Requirements:
- Face clearly visible and centered
- Neutral or friendly expression suitable for video
- Good lighting on face
- Simple, non-distracting background
- Portrait orientation, head and shoulders visible
- High quality, suitable for video animation"""

        return prompt

    def _get_cache_key(self, description: str, style: str) -> str:
        """Generate cache key for avatar"""
        content = f"{description}:{style}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if avatar exists in cache"""
        cache_path = self.cache_dir / f"{cache_key}.png"
        if cache_path.exists():
            return str(cache_path)
        return None

    async def generate_with_openai(self, prompt: str, style: str) -> AvatarResult:
        """Generate avatar using OpenAI DALL-E 3"""
        # Check both config and environment directly
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")

        print(f"\n[OPENAI] Attempting OpenAI DALL-E 3 generation...")
        print(f"[OPENAI] API Key found: {bool(api_key)}")
        if api_key:
            print(f"[OPENAI] API Key starts with: {api_key[:7]}...")

        if not api_key:
            print("[OPENAI] ERROR: No API key configured!")
            return AvatarResult(success=False, error="OpenAI API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "quality": "hd",
                    "response_format": "b64_json"
                }

                print(f"[OPENAI] Sending request to OpenAI API...")
                async with session.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    print(f"[OPENAI] Response status: {response.status}")
                    if response.status == 200:
                        print("[OPENAI] SUCCESS! Image generated!")
                        data = await response.json()
                        image_b64 = data["data"][0]["b64_json"]
                        revised_prompt = data["data"][0].get("revised_prompt", prompt)

                        # Save to cache
                        cache_key = self._get_cache_key(prompt, style)
                        cache_path = self.cache_dir / f"{cache_key}.png"

                        image_data = base64.b64decode(image_b64)
                        with open(cache_path, "wb") as f:
                            f.write(image_data)

                        return AvatarResult(
                            success=True,
                            image_path=str(cache_path),
                            image_base64=image_b64,
                            provider="openai_dalle3",
                            style=style,
                            prompt=prompt,
                            metadata={
                                "revised_prompt": revised_prompt,
                                "model": "dall-e-3",
                                "size": "1024x1024"
                            }
                        )
                    else:
                        error_data = await response.text()
                        print(f"[OPENAI] ERROR: Status {response.status}")
                        print(f"[OPENAI] Error details: {error_data}")
                        return AvatarResult(
                            success=False,
                            error=f"OpenAI API error: {response.status} - {error_data}"
                        )

        except asyncio.TimeoutError:
            print("[OPENAI] ERROR: Request timed out after 120 seconds")
            return AvatarResult(success=False, error="OpenAI request timed out")
        except Exception as e:
            print(f"[OPENAI] ERROR: Exception occurred: {str(e)}")
            return AvatarResult(success=False, error=f"OpenAI error: {str(e)}")

    def generate_placeholder(self, description: str, style: str) -> AvatarResult:
        """Generate a placeholder avatar for testing (no external API needed)"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            # Create image
            size = (512, 512)

            # Generate colors based on description hash
            desc_hash = hash(description)
            random.seed(desc_hash)

            bg_color = (
                random.randint(100, 200),
                random.randint(100, 200),
                random.randint(100, 200)
            )

            face_color = (
                random.randint(200, 255),
                random.randint(180, 220),
                random.randint(160, 200)
            )

            img = Image.new('RGB', size, bg_color)
            draw = ImageDraw.Draw(img)

            # Draw simple avatar shape
            # Face circle
            face_center = (256, 220)
            face_radius = 120
            draw.ellipse(
                [face_center[0] - face_radius, face_center[1] - face_radius,
                 face_center[0] + face_radius, face_center[1] + face_radius],
                fill=face_color, outline=(100, 100, 100), width=2
            )

            # Eyes
            eye_y = 200
            left_eye_x = 210
            right_eye_x = 300
            eye_radius = 15

            # Eye whites
            draw.ellipse(
                [left_eye_x - eye_radius, eye_y - eye_radius,
                 left_eye_x + eye_radius, eye_y + eye_radius],
                fill="white", outline=(100, 100, 100)
            )
            draw.ellipse(
                [right_eye_x - eye_radius, eye_y - eye_radius,
                 right_eye_x + eye_radius, eye_y + eye_radius],
                fill="white", outline=(100, 100, 100)
            )

            # Pupils
            pupil_radius = 7
            draw.ellipse(
                [left_eye_x - pupil_radius, eye_y - pupil_radius,
                 left_eye_x + pupil_radius, eye_y + pupil_radius],
                fill=(50, 50, 50)
            )
            draw.ellipse(
                [right_eye_x - pupil_radius, eye_y - pupil_radius,
                 right_eye_x + pupil_radius, eye_y + pupil_radius],
                fill=(50, 50, 50)
            )

            # Smile
            draw.arc(
                [200, 220, 310, 290],
                start=20, end=160,
                fill=(150, 100, 100), width=3
            )

            # Hair (simple)
            hair_color = (
                random.randint(50, 150),
                random.randint(30, 100),
                random.randint(20, 80)
            )
            draw.ellipse(
                [face_center[0] - face_radius - 10, face_center[1] - face_radius - 40,
                 face_center[0] + face_radius + 10, face_center[1] - 20],
                fill=hair_color
            )

            # Body/shoulders
            draw.polygon(
                [(150, 400), (256, 350), (360, 400), (400, 520), (110, 520)],
                fill=(random.randint(50, 150), random.randint(50, 150), random.randint(100, 200))
            )

            # Add text overlay
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

            # Style badge
            draw.rectangle([(10, 10), (150, 35)], fill=(0, 0, 0, 128))
            draw.text((15, 12), f"Style: {style}", fill="white", font=font)

            # Save image
            cache_key = self._get_cache_key(description, style)
            cache_path = self.cache_dir / f"{cache_key}.png"
            img.save(cache_path, "PNG")

            # Convert to base64
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            return AvatarResult(
                success=True,
                image_path=str(cache_path),
                image_base64=image_b64,
                provider="placeholder",
                style=style,
                prompt=description,
                metadata={
                    "type": "placeholder",
                    "note": "This is a placeholder avatar. Configure AI API keys for real avatars.",
                    "description": description[:100]
                }
            )

        except Exception as e:
            return AvatarResult(success=False, error=f"Placeholder generation error: {str(e)}")

    async def generate(
        self,
        description: str,
        style: str = "realistic",
        use_cache: bool = True
    ) -> AvatarResult:
        """
        Generate an avatar from a text description.

        Args:
            description: Text description of the avatar to create
            style: Visual style (realistic, anime, cartoon, etc.)
            use_cache: Whether to use cached results

        Returns:
            AvatarResult with generated image data
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(description, style)
            cached_path = self._check_cache(cache_key)
            if cached_path:
                # Load cached image
                with open(cached_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode()
                return AvatarResult(
                    success=True,
                    image_path=cached_path,
                    image_base64=image_b64,
                    provider="cache",
                    style=style,
                    prompt=description,
                    metadata={"cached": True}
                )

        # Build optimized prompt
        prompt = self._build_avatar_prompt(description, style)

        # Refresh providers list (in case API keys were loaded later)
        providers = self._init_providers()

        print(f"\n{'='*60}")
        print(f"[AVATAR] Starting avatar generation")
        print(f"[AVATAR] Description: {description[:50]}...")
        print(f"[AVATAR] Style: {style}")
        print(f"[AVATAR] Available providers: {[p['name'] for p in providers]}")
        print(f"{'='*60}")

        # Try providers in order
        for provider in providers:
            print(f"\n[AVATAR] Trying provider: {provider['name']}")
            if not provider.get("enabled"):
                continue

            if provider["name"] == "openai_dalle3":
                result = await self.generate_with_openai(prompt, style)
                if result.success:
                    print(f"[AVATAR] OpenAI succeeded!")
                    return result
                else:
                    print(f"[AVATAR] OpenAI failed: {result.error}")
                    print(f"[AVATAR] Falling back to next provider...")

            elif provider["name"] == "placeholder":
                print(f"[AVATAR] Using placeholder generator (fallback)")
                result = self.generate_placeholder(description, style)
                if result.success:
                    print(f"[AVATAR] Placeholder generated successfully")
                    return result

        return AvatarResult(
            success=False,
            error="All avatar generation providers failed"
        )

    def get_available_styles(self) -> List[str]:
        """Get list of available avatar styles"""
        return self.config.avatar_styles

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "providers": [
                {
                    "name": p["name"],
                    "enabled": p.get("enabled", False),
                    "priority": p.get("priority", 99)
                }
                for p in self.providers
            ],
            "openai_configured": bool(config.openai_api_key),
            "did_configured": bool(config.did_api_key)
        }


# Singleton instance
avatar_generator = AvatarGenerator()
