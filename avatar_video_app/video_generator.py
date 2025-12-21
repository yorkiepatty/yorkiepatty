"""
Video Generator - Creates talking avatar videos with lip-sync
Supports multiple video generation backends
"""
import os
import io
import base64
import json
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import tempfile
import shutil

from .config import config


@dataclass
class VideoResult:
    """Result from video generation"""
    success: bool
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    duration: float = 0.0
    resolution: Optional[tuple] = None
    provider: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    job_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VideoGenerator:
    """
    Generates talking avatar videos from images and audio.
    Uses AI-powered lip-sync technology.
    """

    def __init__(self):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(config.temp_dir) / "video"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.providers = self._init_providers()
        self._jobs = {}  # Track async jobs

    def _init_providers(self) -> List[Dict[str, Any]]:
        """Initialize video generation providers"""
        providers = []

        # Debug: Show config state
        print(f"[VIDEO_GEN] Initializing providers...")
        print(f"[VIDEO_GEN] config.hedra_api_key: {'SET' if config.hedra_api_key else 'NOT SET'}")

        # Hedra API - BEST for animals, cartoons, and non-human avatars
        if config.hedra_api_key:
            print(f"[VIDEO_GEN] Adding Hedra provider (priority 1)")
            providers.append({
                "name": "hedra",
                "enabled": True,
                "priority": 1,
                "endpoint": "https://api.hedra.com/v1",
                "description": "Hedra API - optimized for animals, cartoons, and stylized avatars"
            })
        else:
            print(f"[VIDEO_GEN] Hedra provider NOT added (no API key)")

        # Local generator is fallback - works with ANY avatar (animals, cartoons, etc.)
        print(f"[VIDEO_GEN] Adding Local provider (priority 2)")
        providers.append({
            "name": "local",
            "enabled": True,
            "priority": 2,
            "description": "Local video generator using OpenCV/FFmpeg - works with any avatar type"
        })

        # D-ID API (talking avatars) - only works with HUMAN faces
        # Disabled by default since it doesn't support animals/cartoons
        if config.did_api_key:
            providers.append({
                "name": "did",
                "enabled": False,  # Disabled by default - human faces only
                "priority": 3,
                "endpoint": "https://api.d-id.com/talks",
                "max_duration": 180,  # 3 minutes
                "note": "Only works with human faces - not animals or cartoons"
            })

        sorted_providers = sorted(providers, key=lambda x: x.get("priority", 99))
        print(f"[VIDEO_GEN] Initialized {len(sorted_providers)} providers: {[p['name'] for p in sorted_providers]}")
        return sorted_providers

    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(os.urandom(8)).hexdigest()[:8]
        return f"video_{timestamp}_{random_part}"

    async def generate_video(
        self,
        avatar_image_path: str,
        audio_path: str,
        output_name: Optional[str] = None
    ) -> VideoResult:
        """
        Generate a talking avatar video.

        Args:
            avatar_image_path: Path to avatar image
            audio_path: Path to audio file
            output_name: Optional custom output filename

        Returns:
            VideoResult with video data or job status
        """
        job_id = self._generate_job_id()

        # Validate inputs
        if not os.path.exists(avatar_image_path):
            return VideoResult(
                success=False,
                error=f"Avatar image not found: {avatar_image_path}",
                job_id=job_id
            )

        if not os.path.exists(audio_path):
            return VideoResult(
                success=False,
                error=f"Audio file not found: {audio_path}",
                job_id=job_id
            )

        # Try providers in order
        for provider in self.providers:
            if not provider.get("enabled"):
                continue

            if provider["name"] == "hedra":
                result = await self._generate_with_hedra(
                    avatar_image_path, audio_path, job_id, output_name
                )
                if result.success or result.status == "processing":
                    return result

            elif provider["name"] == "did":
                result = await self._generate_with_did(
                    avatar_image_path, audio_path, job_id, output_name
                )
                if result.success or result.status == "processing":
                    return result

            elif provider["name"] == "local":
                result = await self._generate_local(
                    avatar_image_path, audio_path, job_id, output_name
                )
                if result.success:
                    return result

        return VideoResult(
            success=False,
            error="All video generation providers failed",
            job_id=job_id
        )

    async def _generate_with_did(
        self,
        avatar_path: str,
        audio_path: str,
        job_id: str,
        output_name: Optional[str]
    ) -> VideoResult:
        """Generate video using D-ID API"""
        if not config.did_api_key:
            return VideoResult(success=False, error="D-ID API key not configured")

        try:
            # Read and encode files
            with open(avatar_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

            # Determine image type
            image_ext = Path(avatar_path).suffix.lower()
            image_type = "image/png" if image_ext == ".png" else "image/jpeg"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Basic {config.did_api_key}",
                    "Content-Type": "application/json"
                }

                # Create talk request
                payload = {
                    "source_url": f"data:{image_type};base64,{image_b64}",
                    "script": {
                        "type": "audio",
                        "audio_url": f"data:audio/wav;base64,{audio_b64}"
                    },
                    "config": {
                        "stitch": True,
                        "result_format": "mp4"
                    }
                }

                async with session.post(
                    "https://api.d-id.com/talks",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status in [200, 201]:
                        data = await response.json()
                        did_id = data.get("id")

                        # Store job info
                        self._jobs[job_id] = {
                            "provider": "did",
                            "did_id": did_id,
                            "status": "processing",
                            "output_name": output_name
                        }

                        return VideoResult(
                            success=True,
                            status="processing",
                            job_id=job_id,
                            provider="did",
                            metadata={
                                "did_id": did_id,
                                "message": "Video is being generated. Poll for status."
                            }
                        )
                    else:
                        error_text = await response.text()
                        return VideoResult(
                            success=False,
                            error=f"D-ID API error: {response.status} - {error_text}"
                        )

        except Exception as e:
            return VideoResult(success=False, error=f"D-ID error: {str(e)}")

    async def check_job_status(self, job_id: str) -> VideoResult:
        """Check status of async video generation job"""
        if job_id not in self._jobs:
            return VideoResult(
                success=False,
                error=f"Job not found: {job_id}",
                job_id=job_id
            )

        job = self._jobs[job_id]

        if job["provider"] == "did":
            return await self._check_did_status(job_id, job)

        return VideoResult(
            success=False,
            error=f"Unknown job provider: {job['provider']}",
            job_id=job_id
        )

    async def _check_did_status(self, job_id: str, job: Dict) -> VideoResult:
        """Check D-ID job status"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Basic {config.did_api_key}"
                }

                async with session.get(
                    f"https://api.d-id.com/talks/{job['did_id']}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")

                        if status == "done":
                            result_url = data.get("result_url")

                            # Download video
                            output_name = job.get("output_name") or job_id
                            output_path = self.output_dir / f"{output_name}.mp4"

                            async with session.get(result_url) as video_response:
                                if video_response.status == 200:
                                    video_data = await video_response.read()
                                    with open(output_path, "wb") as f:
                                        f.write(video_data)

                                    # Cleanup job
                                    del self._jobs[job_id]

                                    return VideoResult(
                                        success=True,
                                        status="completed",
                                        video_path=str(output_path),
                                        video_url=result_url,
                                        provider="did",
                                        job_id=job_id,
                                        metadata=data
                                    )

                        elif status in ["created", "started"]:
                            return VideoResult(
                                success=True,
                                status="processing",
                                job_id=job_id,
                                provider="did",
                                metadata={"did_status": status}
                            )

                        elif status == "error":
                            del self._jobs[job_id]
                            return VideoResult(
                                success=False,
                                status="failed",
                                error=data.get("error", "Unknown D-ID error"),
                                job_id=job_id
                            )

                    return VideoResult(
                        success=False,
                        error=f"Failed to check D-ID status: {response.status}"
                    )

        except Exception as e:
            return VideoResult(success=False, error=f"Status check error: {str(e)}")

    async def _generate_with_hedra(
        self,
        avatar_path: str,
        audio_path: str,
        job_id: str,
        output_name: Optional[str]
    ) -> VideoResult:
        """
        Generate video using Hedra API.
        Hedra requires a multi-step process:
        1. Upload image asset → get image_asset_id
        2. Wait for image to be ready
        3. Upload audio asset → get audio_asset_id
        4. Wait for audio to be ready
        5. Request video generation with both asset IDs
        6. Poll for video completion
        7. Download the final video
        """
        if not config.hedra_api_key:
            return VideoResult(success=False, error="Hedra API key not configured")

        try:
            print(f"[HEDRA] Starting multi-step video generation for job {job_id}")

            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-Key": config.hedra_api_key,
                    "Content-Type": "application/json"
                }

                # Step 1: Upload image asset
                print(f"[HEDRA] Step 1: Uploading image asset...")
                with open(avatar_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()

                image_ext = Path(avatar_path).suffix.lower()
                image_type = "image/png" if image_ext == ".png" else "image/jpeg"

                image_payload = {
                    "type": "image",
                    "data": f"data:{image_type};base64,{image_data}"
                }

                async with session.post(
                    "https://api.hedra.com/v1/assets",
                    headers=headers,
                    json=image_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return VideoResult(
                            success=False,
                            error=f"Hedra image upload failed ({response.status}): {error_text}"
                        )

                    image_result = await response.json()
                    image_asset_id = image_result.get("id")
                    print(f"[HEDRA] Image asset uploaded: {image_asset_id}")

                # Step 2: Wait for image to be ready
                print(f"[HEDRA] Step 2: Waiting for image asset to be ready...")
                image_ready = await self._wait_for_hedra_asset(session, headers, image_asset_id, max_wait=60)
                if not image_ready:
                    return VideoResult(
                        success=False,
                        error="Hedra image asset failed to become ready"
                    )
                print(f"[HEDRA] Image asset ready!")

                # Step 3: Upload audio asset
                print(f"[HEDRA] Step 3: Uploading audio asset...")
                with open(audio_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode()

                audio_payload = {
                    "type": "audio",
                    "data": f"data:audio/wav;base64,{audio_data}"
                }

                async with session.post(
                    "https://api.hedra.com/v1/assets",
                    headers=headers,
                    json=audio_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return VideoResult(
                            success=False,
                            error=f"Hedra audio upload failed ({response.status}): {error_text}"
                        )

                    audio_result = await response.json()
                    audio_asset_id = audio_result.get("id")
                    print(f"[HEDRA] Audio asset uploaded: {audio_asset_id}")

                # Step 4: Wait for audio to be ready
                print(f"[HEDRA] Step 4: Waiting for audio asset to be ready...")
                audio_ready = await self._wait_for_hedra_asset(session, headers, audio_asset_id, max_wait=60)
                if not audio_ready:
                    return VideoResult(
                        success=False,
                        error="Hedra audio asset failed to become ready"
                    )
                print(f"[HEDRA] Audio asset ready!")

                # Step 5: Request video generation
                print(f"[HEDRA] Step 5: Requesting video generation...")
                video_payload = {
                    "type": "video",
                    "ai_model_id": "1",  # Default model
                    "start_keyframe_id": image_asset_id,
                    "audio_id": audio_asset_id,
                    "generated_video_inputs": {
                        "resolution": "720p",
                        "aspect_ratio": "9:16",
                        "duration_ms": 8000  # 8 seconds default
                    }
                }

                async with session.post(
                    "https://api.hedra.com/v1/generations",
                    headers=headers,
                    json=video_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return VideoResult(
                            success=False,
                            error=f"Hedra video generation request failed ({response.status}): {error_text}"
                        )

                    generation_result = await response.json()
                    generation_id = generation_result.get("id")
                    print(f"[HEDRA] Video generation started: {generation_id}")

                # Step 6: Poll for video completion
                print(f"[HEDRA] Step 6: Polling for video completion...")
                video_url = await self._wait_for_hedra_video(session, headers, generation_id, max_wait=180)
                if not video_url:
                    return VideoResult(
                        success=False,
                        error="Hedra video generation timed out or failed"
                    )
                print(f"[HEDRA] Video completed: {video_url}")

                # Step 7: Download the video
                print(f"[HEDRA] Step 7: Downloading final video...")
                async with session.get(video_url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        return VideoResult(
                            success=False,
                            error=f"Failed to download Hedra video ({response.status})"
                        )

                    video_data = await response.read()

                    # Save video
                    output_filename = output_name or f"{job_id}_hedra.mp4"
                    output_path = self.output_dir / output_filename

                    with open(output_path, "wb") as f:
                        f.write(video_data)

                    # Encode to base64
                    video_b64 = base64.b64encode(video_data).decode()

                    print(f"[HEDRA] Video saved successfully: {output_path}")

                    return VideoResult(
                        success=True,
                        video_path=str(output_path),
                        video_base64=video_b64,
                        provider="hedra",
                        job_id=job_id,
                        status="completed"
                    )

        except Exception as e:
            print(f"[HEDRA] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return VideoResult(success=False, error=f"Hedra generation error: {str(e)}")

    async def _wait_for_hedra_asset(
        self,
        session: aiohttp.ClientSession,
        headers: dict,
        asset_id: str,
        max_wait: int = 60
    ) -> bool:
        """Wait for Hedra asset to be ready"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            try:
                async with session.get(
                    f"https://api.hedra.com/v1/assets/{asset_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")

                        if status == "ready" or status == "complete":
                            return True
                        elif status == "failed" or status == "error":
                            print(f"[HEDRA] Asset {asset_id} failed: {data.get('error')}")
                            return False

                # Wait before next check
                await asyncio.sleep(2)

            except Exception as e:
                print(f"[HEDRA] Asset check error: {e}")
                await asyncio.sleep(2)

        print(f"[HEDRA] Asset {asset_id} timed out waiting to be ready")
        return False

    async def _wait_for_hedra_video(
        self,
        session: aiohttp.ClientSession,
        headers: dict,
        generation_id: str,
        max_wait: int = 180
    ) -> Optional[str]:
        """Wait for Hedra video generation to complete and return video URL"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            try:
                async with session.get(
                    f"https://api.hedra.com/v1/generations/{generation_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")

                        if status == "complete" or status == "completed":
                            video_url = data.get("video_url") or data.get("url")
                            return video_url
                        elif status == "failed" or status == "error":
                            print(f"[HEDRA] Video generation failed: {data.get('error')}")
                            return None

                # Wait before next check (longer for video generation)
                await asyncio.sleep(5)

            except Exception as e:
                print(f"[HEDRA] Video check error: {e}")
                await asyncio.sleep(5)

        print(f"[HEDRA] Video generation {generation_id} timed out")
        return None

    async def _generate_local(
        self,
        avatar_path: str,
        audio_path: str,
        job_id: str,
        output_name: Optional[str]
    ) -> VideoResult:
        """Generate video locally using OpenCV/FFmpeg"""
        try:
            output_name = output_name or job_id
            output_path = self.output_dir / f"{output_name}.mp4"

            # Check for required libraries
            cv2_available = False
            try:
                import cv2
                cv2_available = True
            except ImportError:
                pass

            # Get audio duration
            audio_duration = await self._get_audio_duration(audio_path)
            if audio_duration <= 0:
                audio_duration = 10  # Default fallback

            if cv2_available:
                result = await self._generate_with_opencv(
                    avatar_path, audio_path, output_path, audio_duration
                )
            else:
                result = await self._generate_with_ffmpeg(
                    avatar_path, audio_path, output_path, audio_duration
                )

            if result and os.path.exists(output_path):
                # Get video info
                file_size = os.path.getsize(output_path)

                with open(output_path, "rb") as f:
                    video_b64 = base64.b64encode(f.read()).decode()

                return VideoResult(
                    success=True,
                    status="completed",
                    video_path=str(output_path),
                    video_base64=video_b64,
                    duration=audio_duration,
                    provider="local",
                    job_id=job_id,
                    metadata={
                        "file_size": file_size,
                        "method": "opencv" if cv2_available else "ffmpeg"
                    }
                )

            return VideoResult(
                success=False,
                error="Local video generation failed",
                job_id=job_id
            )

        except Exception as e:
            return VideoResult(success=False, error=f"Local generation error: {str(e)}")

    async def _generate_with_opencv(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: Path,
        duration: float
    ) -> bool:
        """Generate video using OpenCV with animated talking effects"""
        try:
            import cv2
            import numpy as np

            # Load avatar image
            img = cv2.imread(avatar_path)
            if img is None:
                return False

            # Resize to target resolution (portrait mode for social media)
            target_h, target_w = 1920, 1080
            img = cv2.resize(img, (target_w, target_h))

            # Calculate frames
            fps = config.video_fps
            total_frames = int(duration * fps)

            # Create temp video without audio
            temp_video = str(self.temp_dir / f"temp_{os.path.basename(output_path)}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (target_w, target_h))

            # Try to analyze audio for lip-sync timing
            audio_energy = await self._analyze_audio_energy(audio_path, total_frames)

            # Generate frames with talking animation
            for frame_idx in range(total_frames):
                frame = img.copy()
                t = frame_idx / fps

                # Get audio energy for this frame (simulates when "talking")
                energy = audio_energy[frame_idx] if audio_energy else np.sin(t * 8 * np.pi) * 0.5 + 0.5

                # 1. Subtle breathing/scale animation
                breath_scale = 1.0 + 0.003 * np.sin(t * 1.5 * np.pi)

                # 2. Talking pulse - more prominent when audio is active
                talk_scale = 1.0 + 0.008 * energy * np.sin(t * 12 * np.pi)

                # Combined scale
                scale = breath_scale * talk_scale

                # 3. Slight head bob when talking
                bob_y = int(5 * energy * np.sin(t * 6 * np.pi))

                # Apply transformations
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 0, scale)
                frame = cv2.warpAffine(frame, M, (w, h))

                # 4. Apply head bob translation
                M_translate = np.float32([[1, 0, 0], [0, 1, bob_y]])
                frame = cv2.warpAffine(frame, M_translate, (w, h))

                # 5. Add subtle brightness variation based on talking
                brightness = int(8 * energy * np.sin(t * 10 * np.pi))
                # Clip brightness to valid range to avoid uint8 overflow
                brightness = np.clip(brightness, -30, 30)
                frame = np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

                # 6. Add subtle glow effect when talking (simulates energy/expression)
                if energy > 0.5:
                    # Add warm glow overlay
                    glow = np.zeros_like(frame, dtype=np.uint8)
                    red_val = np.clip(int(20 * (energy - 0.5)), 0, 255)
                    green_val = np.clip(int(10 * (energy - 0.5)), 0, 255)
                    glow[:, :, 2] = red_val  # Red channel
                    glow[:, :, 1] = green_val  # Green channel
                    frame = cv2.addWeighted(frame, 1.0, glow, 0.3, 0)

                out.write(frame)

            out.release()

            # Combine with audio using FFmpeg
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-i", temp_video, "-i", audio_path,
                    "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", "-y", str(output_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()

                # Cleanup temp file
                if os.path.exists(temp_video):
                    os.remove(temp_video)

                return os.path.exists(output_path)

            except Exception:
                # If FFmpeg fails, just use the video without audio
                shutil.move(temp_video, str(output_path))
                return True

        except Exception as e:
            print(f"OpenCV video generation error: {e}")
            return False

    async def _generate_with_ffmpeg(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: Path,
        duration: float
    ) -> bool:
        """Generate video using FFmpeg directly"""
        try:
            # Create video from image + audio
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-loop", "1",
                "-i", avatar_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                "-shortest",
                "-y",
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return os.path.exists(output_path)

        except Exception as e:
            print(f"FFmpeg video generation error: {e}")
            return False

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
        except Exception:
            pass

        # Try FFprobe
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            return float(stdout.decode().strip())
        except Exception:
            pass

        return 0.0

    async def _analyze_audio_energy(self, audio_path: str, num_frames: int) -> Optional[List[float]]:
        """
        Analyze audio to extract energy levels for lip-sync animation.
        Returns a list of energy values (0-1) for each video frame.
        """
        try:
            import numpy as np

            # Try librosa first (best quality)
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=22050, mono=True)

                # Calculate RMS energy
                hop_length = max(1, len(y) // num_frames)
                rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

                # Resample to match video frames
                if len(rms) != num_frames:
                    indices = np.linspace(0, len(rms) - 1, num_frames).astype(int)
                    rms = rms[indices]

                # Normalize to 0-1 range
                rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

                # Apply smoothing for more natural animation
                from scipy.ndimage import gaussian_filter1d
                rms = gaussian_filter1d(rms, sigma=2)

                return rms.tolist()

            except ImportError:
                pass

            # Fallback: Try reading WAV directly
            try:
                import wave
                import struct

                with wave.open(audio_path, 'rb') as wav:
                    n_channels = wav.getnchannels()
                    sampwidth = wav.getsampwidth()
                    framerate = wav.getframerate()
                    n_frames = wav.getnframes()

                    raw_data = wav.readframes(n_frames)

                    # Convert to samples
                    if sampwidth == 2:
                        fmt = f'{n_frames * n_channels}h'
                        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
                    else:
                        samples = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32)

                    # Make mono
                    if n_channels > 1:
                        samples = samples.reshape(-1, n_channels).mean(axis=1)

                    # Calculate energy per video frame
                    samples_per_frame = max(1, len(samples) // num_frames)
                    energy = []
                    for i in range(num_frames):
                        start = i * samples_per_frame
                        end = min(start + samples_per_frame, len(samples))
                        chunk = samples[start:end]
                        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0
                        energy.append(rms)

                    # Normalize
                    energy = np.array(energy)
                    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

                    return energy.tolist()

            except Exception:
                pass

            # Final fallback: Generate synthetic energy pattern
            t = np.linspace(0, num_frames / 30, num_frames)  # Assume 30fps
            # Create varied pattern that simulates speech rhythm
            energy = 0.5 + 0.3 * np.sin(t * 4 * np.pi) + 0.2 * np.sin(t * 7 * np.pi)
            energy = np.clip(energy, 0, 1)
            return energy.tolist()

        except Exception as e:
            print(f"Audio analysis error: {e}")
            return None

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of video generation providers"""
        return {
            "providers": [
                {
                    "name": p["name"],
                    "enabled": p.get("enabled", False),
                    "priority": p.get("priority", 99)
                }
                for p in self.providers
            ],
            "did_configured": bool(config.did_api_key),
            "pending_jobs": len(self._jobs)
        }

    def list_generated_videos(self) -> List[Dict[str, Any]]:
        """List all generated videos"""
        videos = []
        for video_file in self.output_dir.glob("*.mp4"):
            stat = video_file.stat()
            videos.append({
                "name": video_file.name,
                "path": str(video_file),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        return sorted(videos, key=lambda x: x["created"], reverse=True)


# Singleton instance
video_generator = VideoGenerator()
