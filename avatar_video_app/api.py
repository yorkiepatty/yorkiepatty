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
    results = {"steps": []}

    try:
        # Step 1: Generate Avatar
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
        if audio:
            # Use uploaded audio
            audio_data = await audio.read()
            input_format = audio.filename.split(".")[-1].lower() if audio.filename else "wav"

            voice_result = await voice_processor.process_uploaded_audio(
                audio_data=audio_data,
                effect_name=voice_effect,
                input_format=input_format
            )
        elif text:
            # Use TTS
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
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed: {video_result.error}"
            )

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
        raise HTTPException(status_code=500, detail=str(e))


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
