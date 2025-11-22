# Avatar Video Creator

Create talking avatar videos up to 3 minutes long from text descriptions. Choose from multiple avatar styles, record your voice or use text-to-speech, and apply fun voice effects.

## Features

- **Avatar Generation**: Create custom avatars from text descriptions
  - 8 visual styles: Realistic, Anime, Cartoon, 3D Render, Artistic, Pixel Art, Watercolor, Oil Painting
  - AI-powered image generation (OpenAI DALL-E 3 or placeholder mode)
  - Smart caching for faster regeneration

- **Voice Options**:
  - Record voice directly in browser
  - Upload audio files (MP3, WAV, M4A, WebM)
  - Text-to-Speech for typed content

- **Voice Effects**: 10 built-in voice changers
  - Normal, Deep, High, Robot, Echo
  - Whisper, Chipmunk, Villain, Announcer, Ethereal

- **Video Generation**:
  - Up to 3-minute videos
  - Lip-sync animation
  - Multiple output providers (D-ID API or local FFmpeg)
  - Video gallery with download support

- **Yorkie Helper**: Cute animated assistant in the bottom corner
  - Provides navigation tips for each step
  - Lip-sync animation when speaking
  - Customizable with your own Yorkie image!

## Quick Start

### 1. Install Python Dependencies

```bash
pip install fastapi uvicorn aiohttp pillow gtts librosa soundfile numpy scipy
```

### 2. Start the Backend API

```bash
cd /home/user/yorkiepatty
python -m avatar_video_app.run --port 8080
```

### 3. Start the Frontend

```bash
cd /home/user/yorkiepatty/avatar_video_app/frontend
npm install
npm run dev
```

### 4. Add Your Yorkie Image (Optional)

Place your Yorkie picture in the public folder:

```bash
cp /path/to/your/yorkie.png avatar_video_app/frontend/public/yorkie.png
```

The Yorkie helper will use your custom image with lip-sync animation!

### 5. Open the App

Visit `http://localhost:3000` in your browser.

## API Configuration

### Environment Variables

Create a `.env` file or export these variables:

```bash
# Optional - for AI-powered avatar generation
OPENAI_API_KEY=your_openai_key

# Optional - for professional lip-sync video generation
DID_API_KEY=your_did_api_key

# Optional - for voice cloning
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### Without API Keys

The app works without any API keys in "placeholder mode":
- Avatars are generated as simple cartoon representations
- Videos are created using FFmpeg/OpenCV with basic animation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/avatar/generate` | POST | Generate avatar from description |
| `/api/avatar/styles` | GET | List available avatar styles |
| `/api/voice/upload` | POST | Upload and process audio |
| `/api/voice/tts` | POST | Text-to-speech conversion |
| `/api/voice/effects` | GET | List voice effects |
| `/api/video/generate` | POST | Generate video from avatar + audio |
| `/api/video/status/{id}` | GET | Check video generation status |
| `/api/video/list` | GET | List generated videos |
| `/api/generate-full` | POST | Full pipeline (avatar + voice + video) |

## Project Structure

```
avatar_video_app/
├── __init__.py          # Package init
├── config.py            # Configuration settings
├── avatar_generator.py  # Avatar creation module
├── voice_processor.py   # Voice recording & effects
├── video_generator.py   # Video generation with lip-sync
├── api.py               # FastAPI backend
├── run.py               # Run script
├── README.md            # This file
└── frontend/            # React frontend
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        └── components/
            ├── Header.jsx
            ├── AvatarCreator.jsx
            ├── VoiceRecorder.jsx
            ├── VoiceEffects.jsx
            ├── GenerateButton.jsx
            ├── VideoPreview.jsx
            ├── VideoGallery.jsx
            └── YorkieHelper.jsx
```

## Usage Examples

### Generate Avatar via API

```bash
curl -X POST http://localhost:8080/api/avatar/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "A friendly robot with blue eyes", "style": "3d_render"}'
```

### Text-to-Speech

```bash
curl -X POST http://localhost:8080/api/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am your AI avatar!", "effect": "robot"}'
```

### Full Video Pipeline

```bash
curl -X POST http://localhost:8080/api/generate-full \
  -F "avatar_description=A smiling businesswoman" \
  -F "avatar_style=realistic" \
  -F "text=Welcome to our presentation today!" \
  -F "voice_effect=announcer"
```

## Requirements

### Python Dependencies
- Python 3.8+
- FastAPI
- uvicorn
- aiohttp
- Pillow
- gTTS
- librosa (optional, for advanced effects)
- soundfile (optional)
- numpy, scipy (optional)

### Node.js Dependencies
- Node.js 16+
- React 18
- Vite 5
- Tailwind CSS 3
- Axios

### System Dependencies (for local video generation)
- FFmpeg
- OpenCV (optional)

## License

Part of The Christman AI Project
