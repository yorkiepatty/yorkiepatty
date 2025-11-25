"""
Avatar Video App API - FastAPI backend for avatar video generation
"""
import os
import base64
import json
import asyncio
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import config
from .avatar_generator import avatar_generator, AvatarResult
from .voice_processor import voice_processor, VoiceResult
from .video_generator import video_generator, VideoResult


# Conversation job tracking
conversation_jobs = {}


# Create FastAPI app
app = FastAPI(
    title="Avatar Video App",
    description="Create talking avatars from descriptions - 3 min video generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class AvatarRequest(BaseModel):
    """Request to generate an avatar"""
    description: str = Field(..., description="Text description of the avatar to create")
    style: str = Field(default="realistic", description="Visual style of the avatar")
    use_cache: bool = Field(default=True, description="Use cached results if available")


class TTSRequest(BaseModel):
    """Request for text-to-speech"""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="default", description="Voice to use")
    effect: str = Field(default="normal", description="Voice effect to apply")


class VideoRequest(BaseModel):
    """Request to generate a video"""
    avatar_path: str = Field(..., description="Path to avatar image")
    audio_path: str = Field(..., description="Path to audio file")
    output_name: Optional[str] = Field(default=None, description="Custom output filename")


class FullPipelineRequest(BaseModel):
    """Request for full avatar video pipeline"""
    avatar_description: str = Field(..., description="Description of avatar to create")
    avatar_style: str = Field(default="realistic", description="Avatar visual style")
    text: Optional[str] = Field(default=None, description="Text for TTS (if no audio uploaded)")
    voice_effect: str = Field(default="normal", description="Voice effect to apply")
    output_name: Optional[str] = Field(default=None, description="Custom output name")


class StatusResponse(BaseModel):
    """Generic status response"""
    status: str
    message: str
    data: Optional[dict] = None


class APIKeyRequest(BaseModel):
    """Request to set API key"""
    api_key: str = Field(..., description="OpenAI API key")


class ConversationRequest(BaseModel):
    """Request to generate conversation videos"""
    character1: dict = Field(..., description="First character data")
    character2: dict = Field(..., description="Second character data")
    script: List[dict] = Field(..., description="Conversation script with speaker and text")
    mode: str = Field(..., description="Conversation mode: ai or scripted")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Avatar Video App",
        "version": "1.0.0",
        "description": "Create talking avatars from descriptions",
        "max_video_duration": config.max_video_duration,
        "endpoints": {
            "avatar": "/api/avatar/generate",
            "voice": "/api/voice/process",
            "video": "/api/video/generate",
            "full": "/api/generate-full"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "providers": {
            "avatar": avatar_generator.get_provider_status(),
            "video": video_generator.get_provider_status()
        }
    }


# ============== Settings Endpoints ==============

@app.post("/api/settings/apikey")
async def set_api_key(request: APIKeyRequest):
    """Set OpenAI API key from the app"""
    api_key = request.api_key.strip()

    # Validate key format
    if not api_key.startswith("sk-"):
        return {"success": False, "message": "Invalid API key format. Key should start with 'sk-'"}

    # Set in environment and config
    os.environ["OPENAI_API_KEY"] = api_key
    config.openai_api_key = api_key

    print(f"\n[SETTINGS] API Key updated via app!")
    print(f"[SETTINGS] Key starts with: {api_key[:10]}...")

    return {
        "success": True,
        "message": "API key saved! You can now generate AI avatars.",
        "key_preview": f"{api_key[:10]}...{api_key[-4:]}"
    }


@app.get("/api/settings/status")
async def get_settings_status():
    """Check if API keys are configured"""
    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    has_key = bool(api_key) and api_key.startswith("sk-")

    return {
        "openai_configured": has_key,
        "key_preview": f"{api_key[:10]}...{api_key[-4:]}" if has_key else None
    }


# ============== Avatar Endpoints ==============

@app.post("/api/avatar/generate")
async def generate_avatar(request: AvatarRequest):
    """Generate an avatar from text description"""
    result = await avatar_generator.generate(
        description=request.description,
        style=request.style,
        use_cache=request.use_cache
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "success": True,
        "image_path": result.image_path,
        "image_base64": result.image_base64,
        "provider": result.provider,
        "style": result.style,
        "metadata": result.metadata
    }


@app.get("/api/avatar/styles")
async def get_avatar_styles():
    """Get available avatar styles"""
    return {
        "styles": avatar_generator.get_available_styles(),
        "default": "realistic"
    }


@app.post("/api/avatar/upload")
async def upload_avatar(image: UploadFile = File(...)):
    """Upload an existing image to use as avatar"""
    import base64
    from pathlib import Path

    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read the image
    image_data = await image.read()

    # Save to avatar outputs
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(image.filename).suffix if image.filename else ".png"
    filename = f"uploaded_avatar_{timestamp}{ext}"
    output_path = output_dir / filename

    # Save the file
    with open(output_path, "wb") as f:
        f.write(image_data)

    # Encode as base64 for preview
    image_b64 = base64.b64encode(image_data).decode()

    print(f"[AVATAR] Uploaded image saved to: {output_path}")

    return {
        "success": True,
        "image_path": str(output_path),
        "image_base64": image_b64,
        "filename": filename
    }


# ============== Voice Endpoints ==============

@app.post("/api/voice/upload")
async def upload_voice(
    audio: UploadFile = File(...),
    effect: str = Form(default="normal")
):
    """Upload and process voice audio"""
    # Read uploaded file
    audio_data = await audio.read()

    # Determine format from filename
    input_format = audio.filename.split(".")[-1].lower() if audio.filename else "wav"

    result = await voice_processor.process_uploaded_audio(
        audio_data=audio_data,
        effect_name=effect,
        input_format=input_format
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    # Validate duration
    valid, msg = voice_processor.validate_audio_duration(result.duration)

    return {
        "success": True,
        "audio_path": result.audio_path,
        "audio_base64": result.audio_base64,
        "duration": result.duration,
        "effect_applied": result.effect_applied,
        "duration_valid": valid,
        "duration_message": msg,
        "metadata": result.metadata
    }


@app.post("/api/voice/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    result = await voice_processor.text_to_speech(
        text=request.text,
        voice=request.voice,
        effect_name=request.effect
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    # Validate duration
    valid, msg = voice_processor.validate_audio_duration(result.duration)

    return {
        "success": True,
        "audio_path": result.audio_path,
        "audio_base64": result.audio_base64,
        "duration": result.duration,
        "effect_applied": result.effect_applied,
        "duration_valid": valid,
        "duration_message": msg,
        "metadata": result.metadata
    }


@app.get("/api/voice/effects")
async def get_voice_effects():
    """Get available voice effects"""
    return {
        "effects": voice_processor.get_available_effects(),
        "default": "normal"
    }


# ============== Video Endpoints ==============

@app.post("/api/video/generate")
async def generate_video(request: VideoRequest):
    """Generate video from avatar and audio"""
    result = await video_generator.generate_video(
        avatar_image_path=request.avatar_path,
        audio_path=request.audio_path,
        output_name=request.output_name
    )

    if not result.success and result.status != "processing":
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "success": result.success,
        "status": result.status,
        "job_id": result.job_id,
        "video_path": result.video_path,
        "video_base64": result.video_base64,
        "duration": result.duration,
        "provider": result.provider,
        "metadata": result.metadata
    }


@app.get("/api/video/status/{job_id}")
async def check_video_status(job_id: str):
    """Check status of video generation job"""
    result = await video_generator.check_job_status(job_id)

    return {
        "success": result.success,
        "status": result.status,
        "job_id": result.job_id,
        "video_path": result.video_path,
        "video_url": result.video_url,
        "error": result.error,
        "metadata": result.metadata
    }


@app.get("/api/video/list")
async def list_videos():
    """List all generated videos"""
    videos = video_generator.list_generated_videos()
    return {"videos": videos, "count": len(videos)}


@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    """Download a generated video"""
    video_path = Path(config.output_dir) / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename
    )


# ============== Full Pipeline Endpoint ==============

@app.post("/api/generate-full")
async def generate_full_video(
    avatar_description: str = Form(...),
    avatar_style: str = Form(default="realistic"),
    voice_effect: str = Form(default="normal"),
    output_name: str = Form(default=None),
    text: str = Form(default=None),
    audio: UploadFile = File(default=None)
):
    """
    Full pipeline: Generate avatar + process voice + create video

    Either provide:
    - audio: Upload audio file
    - text: Use text-to-speech
    """
    import traceback
    results = {"steps": []}

    try:
        print("\n[PIPELINE] ============================================")
        print("[PIPELINE] Starting full video generation...")
        print(f"[PIPELINE] Received parameters:")
        print(f"[PIPELINE]   avatar_description: '{avatar_description[:50] if avatar_description else 'EMPTY'}'...")
        print(f"[PIPELINE]   avatar_style: '{avatar_style}'")
        print(f"[PIPELINE]   voice_effect: '{voice_effect}'")
        print(f"[PIPELINE]   text: '{text[:50] if text else 'NONE/EMPTY'}' (type={type(text).__name__}, len={len(text) if text else 0})")
        print(f"[PIPELINE]   audio: '{audio.filename if audio else 'NONE'}'")
        print(f"[PIPELINE] Audio source check: text={bool(text)}, audio={bool(audio)}")
        print("[PIPELINE] ============================================")

        # Step 1: Generate Avatar
        print("[PIPELINE] Step 1: Generating avatar...")
        avatar_result = await avatar_generator.generate(
            description=avatar_description,
            style=avatar_style
        )

        if not avatar_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Avatar generation failed: {avatar_result.error}"
            )

        results["steps"].append({
            "step": "avatar",
            "success": True,
            "image_path": avatar_result.image_path
        })

        # Step 2: Process Voice
        print("[PIPELINE] Step 2: Processing voice...")
        if audio:
            # Use uploaded audio
            print(f"[PIPELINE] Using uploaded audio: {audio.filename}")
            audio_data = await audio.read()
            input_format = audio.filename.split(".")[-1].lower() if audio.filename else "wav"

            voice_result = await voice_processor.process_uploaded_audio(
                audio_data=audio_data,
                effect_name=voice_effect,
                input_format=input_format
            )
        elif text:
            # Use TTS
            print(f"[PIPELINE] Using TTS with text length: {len(text)}")
            voice_result = await voice_processor.text_to_speech(
                text=text,
                effect_name=voice_effect
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'audio' file or 'text' must be provided"
            )

        if not voice_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Voice processing failed: {voice_result.error}"
            )

        # Validate duration
        valid, msg = voice_processor.validate_audio_duration(voice_result.duration)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)

        results["steps"].append({
            "step": "voice",
            "success": True,
            "audio_path": voice_result.audio_path,
            "duration": voice_result.duration,
            "effect": voice_result.effect_applied
        })

        # Step 3: Generate Video
        video_result = await video_generator.generate_video(
            avatar_image_path=avatar_result.image_path,
            audio_path=voice_result.audio_path,
            output_name=output_name
        )

        if not video_result.success and video_result.status != "processing":
            # Return error with partial results so user can see what happened
            return {
                "success": False,
                "error": f"Video generation failed: {video_result.error}",
                "avatar": {
                    "image_path": avatar_result.image_path,
                    "image_base64": avatar_result.image_base64,
                    "style": avatar_result.style
                },
                "voice": {
                    "audio_path": voice_result.audio_path,
                    "duration": voice_result.duration
                },
                "video": {
                    "status": "failed",
                    "error": video_result.error
                }
            }

        results["steps"].append({
            "step": "video",
            "success": True,
            "status": video_result.status,
            "job_id": video_result.job_id,
            "video_path": video_result.video_path
        })

        return {
            "success": True,
            "message": "Video generation complete" if video_result.status == "completed" else "Video processing",
            "avatar": {
                "image_path": avatar_result.image_path,
                "image_base64": avatar_result.image_base64,
                "style": avatar_result.style
            },
            "voice": {
                "audio_path": voice_result.audio_path,
                "duration": voice_result.duration,
                "effect": voice_result.effect_applied
            },
            "video": {
                "status": video_result.status,
                "job_id": video_result.job_id,
                "video_path": video_result.video_path,
                "video_base64": video_result.video_base64
            },
            "steps": results["steps"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[PIPELINE] ERROR: {str(e)}")
        print("[PIPELINE] Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== Conversation Endpoints ==============

@app.post("/api/conversation/generate")
async def generate_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Generate videos for a conversation between two avatars"""
    try:
        # Generate unique job ID
        job_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

        # Initialize job tracking
        conversation_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "total_lines": len(request.script),
            "videos": [],
            "errors": [],
            "started_at": datetime.now().isoformat()
        }

        # Start background task to generate videos
        background_tasks.add_task(
            generate_conversation_videos,
            job_id,
            request.character1,
            request.character2,
            request.script,
            request.mode
        )

        return {
            "success": True,
            "job_id": job_id,
            "message": f"Conversation generation started ({len(request.script)} lines)",
            "total_lines": len(request.script)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/status/{job_id}")
async def get_conversation_status(job_id: str):
    """Get status of conversation generation"""
    if job_id not in conversation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = conversation_jobs[job_id]

    return {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total_lines": job["total_lines"],
        "videos": job["videos"],
        "errors": job["errors"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at")
    }


async def generate_conversation_videos(
    job_id: str,
    character1: dict,
    character2: dict,
    script: List[dict],
    mode: str
):
    """Background task to generate all conversation videos"""
    try:
        print(f"[CONVERSATION] Starting generation for job {job_id}")
        print(f"[CONVERSATION] {len(script)} lines to generate")

        # Decode avatar images from base64 if needed
        char1_image_path = await save_character_avatar(character1, "char1")
        char2_image_path = await save_character_avatar(character2, "char2")

        # Generate videos for each line
        for index, line in enumerate(script):
            try:
                print(f"[CONVERSATION] Generating line {index + 1}/{len(script)}: {line['text'][:50]}...")

                # Determine which character is speaking
                speaker_num = line['speaker']
                character = character1 if speaker_num == 1 else character2
                avatar_path = char1_image_path if speaker_num == 1 else char2_image_path

                # Generate TTS audio for this line
                print(f"[CONVERSATION] Generating TTS for line {index + 1}...")
                voice_result = await voice_processor.generate_tts(
                    text=line['text'],
                    voice=character.get('voice', 'default')
                )

                if not voice_result.success:
                    raise Exception(f"TTS failed: {voice_result.error}")

                # Generate video for this line
                print(f"[CONVERSATION] Generating video for line {index + 1}...")
                video_result = await video_generator.generate_video(
                    avatar_image_path=avatar_path,
                    audio_path=voice_result.audio_path,
                    output_name=f"{job_id}_line_{index}"
                )

                # Handle async video generation (HeyGen, Hedra, etc.)
                if video_result.status == "processing":
                    # Poll until complete
                    max_polls = 120  # 10 minutes
                    poll_count = 0
                    while poll_count < max_polls:
                        await asyncio.sleep(5)
                        video_result = await video_generator.check_job_status(video_result.job_id)
                        if video_result.status == "completed":
                            break
                        elif video_result.status == "failed":
                            raise Exception(f"Video generation failed: {video_result.error}")
                        poll_count += 1

                    if poll_count >= max_polls:
                        raise Exception("Video generation timed out")

                # Store video result
                conversation_jobs[job_id]["videos"].append({
                    "index": index,
                    "speaker": speaker_num,
                    "text": line['text'],
                    "video_path": video_result.video_path,
                    "video_base64": video_result.video_base64,
                    "provider": video_result.provider
                })

                # Update progress
                conversation_jobs[job_id]["progress"] = index + 1
                print(f"[CONVERSATION] Completed line {index + 1}/{len(script)}")

            except Exception as e:
                error_msg = f"Line {index + 1} failed: {str(e)}"
                print(f"[CONVERSATION] ERROR: {error_msg}")
                conversation_jobs[job_id]["errors"].append(error_msg)

        # Mark as completed
        conversation_jobs[job_id]["status"] = "completed"
        conversation_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        print(f"[CONVERSATION] Job {job_id} completed successfully")

    except Exception as e:
        print(f"[CONVERSATION] Job {job_id} failed: {str(e)}")
        conversation_jobs[job_id]["status"] = "failed"
        conversation_jobs[job_id]["errors"].append(str(e))
        conversation_jobs[job_id]["completed_at"] = datetime.now().isoformat()


async def save_character_avatar(character: dict, prefix: str) -> str:
    """Save character avatar image and return path"""
    avatar_image = character.get('avatarImage', '')

    if not avatar_image:
        # No avatar image provided, use a default or raise error
        raise Exception(f"Character {character.get('name', 'unknown')} has no avatar image")

    # If it's a base64 data URL
    if avatar_image.startswith('data:image'):
        # Extract base64 data
        header, base64_data = avatar_image.split(',', 1)
        image_data = base64.b64decode(base64_data)

        # Determine file extension from header
        if 'png' in header:
            ext = 'png'
        elif 'jpeg' in header or 'jpg' in header:
            ext = 'jpg'
        else:
            ext = 'png'

        # Save to temp file
        temp_dir = Path(config.temp_dir) / "conversation"
        temp_dir.mkdir(parents=True, exist_ok=True)

        avatar_path = temp_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        with open(avatar_path, 'wb') as f:
            f.write(image_data)

        return str(avatar_path)

    # If it's already a file path
    elif os.path.exists(avatar_image):
        return avatar_image

    else:
        raise Exception(f"Invalid avatar image for character {character.get('name', 'unknown')}")


# ============== Static Files (for frontend) ==============

# Mount output directory for video access
app.mount("/outputs", StaticFiles(directory=config.output_dir), name="outputs")


# ============== Run App ==============

def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
