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

        # Local generator is primary - works with ANY avatar (animals, cartoons, etc.)
        providers.append({
            "name": "local",
            "enabled": True,
            "priority": 1,
            "description": "Local video generator using OpenCV/FFmpeg - works with any avatar type"
        })

        # D-ID API (talking avatars) - only works with HUMAN faces
        # Disabled by default since it doesn't support animals/cartoons
        if config.did_api_key:
            providers.append({
                "name": "did",
                "enabled": False,  # Disabled by default - human faces only
                "priority": 2,
                "endpoint": "https://api.d-id.com/talks",
                "max_duration": 180,  # 3 minutes
                "note": "Only works with human faces - not animals or cartoons"
            })

        return sorted(providers, key=lambda x: x.get("priority", 99))

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

            if provider["name"] == "did":
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

    async def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        # Check common FFmpeg paths on Windows
        ffmpeg_paths = [
            "ffmpeg",  # System PATH
            r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe",  # User's install location
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]

        for ffmpeg_cmd in ffmpeg_paths:
            try:
                process = await asyncio.create_subprocess_exec(
                    ffmpeg_cmd, "-version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
                if process.returncode == 0:
                    self._ffmpeg_path = ffmpeg_cmd  # Store for later use
                    print(f"[VIDEO] Found FFmpeg at: {ffmpeg_cmd}")
                    return True
            except FileNotFoundError:
                continue
            except Exception:
                continue
        return False

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

            # Check for FFmpeg
            ffmpeg_available = await self._check_ffmpeg()

            print(f"\n[VIDEO] OpenCV available: {cv2_available}")
            print(f"[VIDEO] FFmpeg available: {ffmpeg_available}")

            if not ffmpeg_available:
                return VideoResult(
                    success=False,
                    error="FFmpeg not found! Please install FFmpeg: https://ffmpeg.org/download.html (On Windows: download, extract, add to PATH)",
                    job_id=job_id
                )

            # Get audio duration
            audio_duration = await self._get_audio_duration(audio_path)
            if audio_duration <= 0:
                audio_duration = 10  # Default fallback

            print(f"[VIDEO] Audio duration: {audio_duration}s")
            print(f"[VIDEO] Generating video...")

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
                print(f"[VIDEO] SUCCESS! Video created: {output_path} ({file_size} bytes)")

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

            print(f"[VIDEO] FAILED: Video file not created")
            return VideoResult(
                success=False,
                error="Video generation failed - check if OpenCV and FFmpeg are properly installed",
                job_id=job_id
            )

        except Exception as e:
            print(f"[VIDEO] ERROR: {str(e)}")
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
                print(f"[VIDEO] Failed to load image: {avatar_path}")
                return False

            # Resize to target resolution (portrait mode for social media)
            target_h, target_w = 1920, 1080
            img = cv2.resize(img, (target_w, target_h))

            # Calculate frames
            fps = config.video_fps
            total_frames = int(duration * fps)

            print(f"[VIDEO] Generating {total_frames} frames at {fps} fps...")

            # Create temp video without audio
            temp_video = str(self.temp_dir / f"temp_{os.path.basename(output_path)}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (target_w, target_h))

            # Try to analyze audio for lip-sync timing
            audio_energy = await self._analyze_audio_energy(audio_path, total_frames)

            if audio_energy:
                max_energy = max(audio_energy)
                min_energy = min(audio_energy)
                print(f"[VIDEO] Audio energy range: {min_energy:.3f} - {max_energy:.3f}")
            else:
                print(f"[VIDEO] WARNING: No audio energy data, using sine wave simulation")

            # Image dimensions
            h, w = img.shape[:2]

            # Define regions for animation (works for faces and animals)
            # Mouth region: lower third, center
            mouth_y_start = int(h * 0.55)
            mouth_y_end = int(h * 0.75)
            mouth_x_start = int(w * 0.3)
            mouth_x_end = int(w * 0.7)

            # Eye region: upper third
            eye_y_start = int(h * 0.25)
            eye_y_end = int(h * 0.45)

            print(f"[VIDEO] Mouth region: y={mouth_y_start}-{mouth_y_end}, x={mouth_x_start}-{mouth_x_end}")

            # Blink timing (random blinks every 2-5 seconds)
            blink_frames = []
            blink_time = 0
            while blink_time < duration:
                blink_time += np.random.uniform(2.0, 5.0)
                if blink_time < duration:
                    blink_frame = int(blink_time * fps)
                    # Add blink duration (about 0.15 seconds)
                    for bf in range(blink_frame, min(blink_frame + int(fps * 0.15), total_frames)):
                        blink_frames.append(bf)

            print(f"[VIDEO] Scheduled {len(set(blink_frames))} blink frames")

            # Generate frames with talking animation
            for frame_idx in range(total_frames):
                frame = img.copy()
                t = frame_idx / fps

                # Get audio energy for this frame
                if audio_energy and frame_idx < len(audio_energy):
                    energy = audio_energy[frame_idx]
                else:
                    energy = abs(np.sin(t * 6 * np.pi)) * 0.8

                # === 1. MOUTH MOVEMENT - Create lip separation effect ===
                if energy > 0.1:
                    # Extract mouth region
                    mouth_region = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end].copy()
                    mr_h, mr_w = mouth_region.shape[:2]

                    if mr_h > 20 and mr_w > 20:
                        # Calculate how much to open the mouth (more energy = more open)
                        open_pixels = int(mr_h * energy * 0.4)  # Up to 40% opening

                        if open_pixels > 3:
                            # Create mouth opening by pulling lips apart from center
                            lip_line = mr_h // 2  # Where the lips meet

                            # Create destination map for remapping
                            map_y = np.zeros((mr_h, mr_w), dtype=np.float32)
                            map_x = np.zeros((mr_h, mr_w), dtype=np.float32)

                            for y in range(mr_h):
                                for x in range(mr_w):
                                    map_x[y, x] = x

                                    # Calculate distance from lip line
                                    dist_from_lip = y - lip_line

                                    # Create opening effect - push pixels away from center
                                    if dist_from_lip < 0:  # Above lip line - push up
                                        # More push near the center, less at edges
                                        center_factor = 1.0 - abs(x - mr_w/2) / (mr_w/2)
                                        push = -open_pixels * 0.5 * center_factor * (1 - abs(dist_from_lip) / lip_line)
                                        map_y[y, x] = y + push
                                    else:  # Below lip line - push down
                                        center_factor = 1.0 - abs(x - mr_w/2) / (mr_w/2)
                                        push = open_pixels * 0.5 * center_factor * (1 - dist_from_lip / (mr_h - lip_line))
                                        map_y[y, x] = y + push

                            # Apply the remap
                            opened_mouth = cv2.remap(mouth_region, map_x, map_y, cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REFLECT)

                            # Add dark interior where mouth opens (center band)
                            gap_center = lip_line
                            gap_size = int(open_pixels * 0.3)
                            if gap_size > 1:
                                for gy in range(max(0, gap_center - gap_size), min(mr_h, gap_center + gap_size)):
                                    # Fade the darkness towards edges
                                    darkness = 1.0 - abs(gy - gap_center) / gap_size
                                    for gx in range(mr_w):
                                        # More dark in center, less at sides
                                        x_factor = 1.0 - (abs(gx - mr_w/2) / (mr_w/2)) ** 2
                                        blend = darkness * x_factor * 0.6
                                        opened_mouth[gy, gx] = (
                                            opened_mouth[gy, gx] * (1 - blend) +
                                            np.array([30, 20, 40]) * blend
                                        ).astype(np.uint8)

                            # Create smooth blend mask (elliptical, centered)
                            mask = np.zeros((mr_h, mr_w), dtype=np.float32)
                            cv2.ellipse(mask, (mr_w//2, mr_h//2), (mr_w//2 - 5, mr_h//2 - 5),
                                       0, 0, 360, 1.0, -1)
                            mask = cv2.GaussianBlur(mask, (21, 21), 0)
                            mask_3ch = np.stack([mask, mask, mask], axis=-1)

                            # Blend with original
                            blended = (opened_mouth.astype(float) * mask_3ch +
                                       mouth_region.astype(float) * (1 - mask_3ch))
                            frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = blended.astype(np.uint8)

                # === 2. EYE BLINKING ===
                if frame_idx in blink_frames:
                    # Squish the eye region vertically to simulate blink
                    eye_region = frame[eye_y_start:eye_y_end, :].copy()
                    er_h = eye_region.shape[0]

                    # Calculate blink progress (0 = open, 1 = closed)
                    blink_start = min([bf for bf in blink_frames if bf <= frame_idx and bf >= frame_idx - int(fps * 0.15)], default=frame_idx)
                    blink_progress = (frame_idx - blink_start) / max(1, int(fps * 0.075))
                    if blink_progress > 1:
                        blink_progress = 2 - blink_progress  # Opening back up
                    blink_progress = max(0, min(1, blink_progress))

                    # Squish factor (1 = normal, 0.3 = almost closed)
                    squish = 1.0 - (blink_progress * 0.6)
                    new_h = max(10, int(er_h * squish))

                    # Resize and pad
                    squished = cv2.resize(eye_region, (eye_region.shape[1], new_h))
                    pad_top = (er_h - new_h) // 2
                    pad_bottom = er_h - new_h - pad_top

                    if pad_top > 0 or pad_bottom > 0:
                        squished = cv2.copyMakeBorder(squished, pad_top, pad_bottom, 0, 0,
                                                      cv2.BORDER_REFLECT)
                    frame[eye_y_start:eye_y_end, :] = squished[:er_h, :]

                # === 3. SUBTLE HEAD MOVEMENT ===
                # Small scale breathing
                breath = 1.0 + 0.008 * np.sin(t * 1.2 * np.pi)

                # Small movement when talking
                move_x = int(4 * energy * np.sin(t * 5 * np.pi))
                move_y = int(3 * energy * np.sin(t * 7 * np.pi))

                # Apply transformations
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 0, breath)
                M[0, 2] += move_x
                M[1, 2] += move_y
                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

                out.write(frame)

                # Progress indicator
                if frame_idx % 100 == 0:
                    print(f"[VIDEO] Progress: {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%) energy={energy:.2f}")

            out.release()
            print(f"[VIDEO] Frames generated, combining with audio...")

            # Combine with audio using FFmpeg
            try:
                ffmpeg_cmd = getattr(self, '_ffmpeg_path', 'ffmpeg')
                process = await asyncio.create_subprocess_exec(
                    ffmpeg_cmd, "-i", temp_video, "-i", audio_path,
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
            # Use stored FFmpeg path or default
            ffmpeg_cmd = getattr(self, '_ffmpeg_path', 'ffmpeg')
            print(f"[VIDEO] Using FFmpeg: {ffmpeg_cmd}")

            # Create video from image + audio
            process = await asyncio.create_subprocess_exec(
                ffmpeg_cmd,
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

            if os.path.exists(output_path):
                return True
            else:
                print(f"[VIDEO] FFmpeg stderr: {stderr.decode()[:500] if stderr else 'No output'}")
                return False

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

        # Try FFprobe - derive path from ffmpeg path
        ffprobe_paths = ["ffprobe"]
        if hasattr(self, '_ffmpeg_path') and self._ffmpeg_path:
            # Derive ffprobe path from ffmpeg path
            ffprobe_paths.insert(0, self._ffmpeg_path.replace("ffmpeg", "ffprobe"))

        for ffprobe_cmd in ffprobe_paths:
            try:
                process = await asyncio.create_subprocess_exec(
                    ffprobe_cmd, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                return float(stdout.decode().strip())
            except Exception:
                continue

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
