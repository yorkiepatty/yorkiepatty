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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
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
async def upload_avatar(file: UploadFile = File(...)):
    """Upload an existing avatar image"""
    import base64
    from pathlib import Path
    import uuid

    try:
        # Check file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read file data
        image_data = await file.read()

        # Save to output directory
        file_ext = file.filename.split('.')[-1] if file.filename else 'png'
        avatar_id = str(uuid.uuid4())[:8]
        save_path = Path(config.output_dir) / f"uploaded_{avatar_id}.{file_ext}"

        with open(save_path, 'wb') as f:
            f.write(image_data)

        # Encode as base64 for preview
        image_base64 = base64.b64encode(image_data).decode()

        return {
            "success": True,
            "image_path": str(save_path),
            "image_base64": image_base64,
            "provider": "upload",
            "metadata": {
                "filename": file.filename,
                "size": len(image_data)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/avatars/list")
async def list_avatars():
    """List all available avatars from avatar_outputs directory"""
    import os
    import base64
    from pathlib import Path

    avatars = []
    avatar_dir = Path("avatar_outputs")

    if not avatar_dir.exists():
        return {"success": True, "avatars": []}

    # Get all image files
    for file_path in avatar_dir.glob("*"):
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            try:
                # Read and encode image
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = f"data:image/{file_path.suffix[1:]};base64,{base64.b64encode(image_data).decode()}"

                avatars.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "source": "local",
                    "image_base64": image_base64,
                    "is_heygen": False,
                    "heygen_id": None,
                    "heygen_name": None
                })
            except Exception as e:
                print(f"Error loading avatar {file_path}: {e}")
                continue

    return {
        "success": True,
        "avatars": avatars
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
    avatar_description: str = Form(default=None),
    avatar_style: str = Form(default="realistic"),
    avatar_path: str = Form(default=None),
    voice_effect: str = Form(default="normal"),
    output_name: str = Form(default=None),
    text: str = Form(default=None),
    voice: str = Form(default="male"),
    audio: UploadFile = File(default=None)
):
    """
    Full pipeline: Generate avatar + process voice + create video

    Either provide:
    - avatar_path: Use existing avatar image
    - avatar_description: Generate new avatar

    And either:
    - audio: Upload audio file
    - text: Use text-to-speech
    """
    results = {"steps": []}

    try:
        # Step 1: Get or Generate Avatar
        if avatar_path:
            # Use existing avatar
            avatar_image_path = avatar_path
            results["steps"].append({
                "step": "avatar",
                "success": True,
                "image_path": avatar_image_path,
                "note": "Using existing avatar"
            })
        elif avatar_description:
            # Generate new avatar
            avatar_result = await avatar_generator.generate(
                description=avatar_description,
                style=avatar_style
            )

            if not avatar_result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Avatar generation failed: {avatar_result.error}"
                )

            avatar_image_path = avatar_result.image_path
            results["steps"].append({
                "step": "avatar",
                "success": True,
                "image_path": avatar_image_path
            })
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'avatar_path' or 'avatar_description' must be provided"
            )

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
                voice=voice,
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
            avatar_image_path=avatar_image_path,
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


# ============== Conversation Generation ==============

@app.post("/api/conversation/generate")
async def generate_conversation(request: Request):
    """Generate a multi-character conversation video"""
    try:
        data = await request.json()

        character1 = data.get('character1')
        character2 = data.get('character2')
        script = data.get('script', [])
        mode = data.get('mode', 'scripted')

        if not character1 or not character2:
            raise HTTPException(status_code=400, detail="Both characters are required")

        if not script or len(script) == 0:
            raise HTTPException(status_code=400, detail="Script is required")

        job_id = video_generator._generate_job_id()
        videos = []

        print(f"[CONVERSATION] Starting conversation generation")
        print(f"[CONVERSATION] Character 1: {character1.get('name')} - avatarType: {character1.get('avatarType', 'NOT SET')}")
        print(f"[CONVERSATION] Character 2: {character2.get('name')} - avatarType: {character2.get('avatarType', 'NOT SET')}")
        print(f"[CONVERSATION] Script lines: {len(script)}")

        # Generate each line
        for index, line in enumerate(script):
            speaker_num = line.get('speaker')
            text = line.get('text')

            if not text:
                continue

            # Select the character
            character = character1 if speaker_num == 1 else character2
            avatar_type = character.get('avatarType', 'human')

            print(f"[CONVERSATION] Generating line {index + 1}/{len(script)}")
            print(f"[CONVERSATION] Speaker: {character.get('name')}")
            print(f"[CONVERSATION] Avatar type: {avatar_type}")
            print(f"[CONVERSATION] Text: {text}")

            # Get avatar image path or base64
            avatar_path = character.get('path')
            if not avatar_path and character.get('avatarImage'):
                # If no path, save the base64 image temporarily
                import base64
                import tempfile
                from pathlib import Path

                avatar_data = character.get('avatarImage')
                if avatar_data.startswith('data:image'):
                    avatar_data = avatar_data.split(',')[1]

                # Save to temp file
                temp_dir = Path(config.output_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)
                avatar_path = str(temp_dir / f"{job_id}_speaker{speaker_num}.png")

                with open(avatar_path, 'wb') as f:
                    f.write(base64.b64decode(avatar_data))

            # Generate voice (TTS)
            voice_result = await voice_processor.text_to_speech(
                text=text,
                voice="default",
                effect_name="normal"
            )

            if not voice_result.success:
                raise HTTPException(status_code=500, detail=f"Voice generation failed: {voice_result.error}")

            # Route to appropriate provider based on avatar type
            # HeyGen for humans, Hedra for animals/cartoons
            preferred_provider = "heygen" if avatar_type == "human" else "hedra"
            print(f"[CONVERSATION] Generating video for {avatar_type} avatar using {preferred_provider}...")

            video_result = await video_generator.generate_video(
                avatar_image_path=avatar_path,
                audio_path=voice_result.audio_path,
                output_name=f"{job_id}_line_{index}",
                preferred_provider=preferred_provider
            )

            if not video_result.success:
                raise HTTPException(status_code=500, detail=f"Video generation failed: {video_result.error}")

            # Add to results
            videos.append({
                "speaker": speaker_num,
                "text": text,
                "character_name": character.get('name'),
                "video_base64": video_result.video_base64,
                "video_path": video_result.video_path,
                "duration": voice_result.duration,
                "provider": video_result.provider
            })

            print(f"[CONVERSATION] Line {index + 1} generated successfully with {video_result.provider}")

        return {
            "success": True,
            "videos": videos,
            "job_id": job_id,
            "total_lines": len(videos)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[CONVERSATION] Error: {str(e)}")
        import traceback
        traceback.print_exc()
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
