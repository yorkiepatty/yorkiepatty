# Royce AI Assistant

A powerful AI assistant for Windows 11 that talks like an old friend.

Royce is confident, smart, and genuinely caring. He doesn't use asterisks or describe actions — he just talks to you like a real person.

## What Royce Can Do

- **Play music** — "Play some Kendrick Lamar" (YouTube via yt-dlp)
- **Deep research** — "Who is Nikola Tesla?" or "Deep dive into quantum computing"
- **Remember everything** — Names, birthdays, preferences, past conversations. Always.
- **Daily horoscopes** — "What's my Scorpio horoscope?"
- **Breaking news** — "What's happening in the world?" with auto-refresh
- **Read your mood** — Notices when you're down, excited, or stressed and responds naturally
- **AI images** — "Make me a picture of a sunset over the ocean" (DALL-E or Stable Diffusion)
- **Patient listener** — Gives you time to finish your thought before responding

## Quick Install (Windows 11)

1. Download this repo as a ZIP
2. Unzip it anywhere
3. Double-click **`install_royce.bat`**
4. Follow the prompts (it'll ask for your API keys)
5. Double-click the **Royce** shortcut on your desktop

That's it.

## API Keys You'll Need

| Key | What it powers | Where to get it |
|-----|---------------|-----------------|
| **ANTHROPIC_API_KEY** | Royce's brain (required) | [console.anthropic.com](https://console.anthropic.com) |
| **ELEVENLABS_API_KEY** | Royce's voice (recommended) | [elevenlabs.io](https://elevenlabs.io) |
| PERPLEXITY_API_KEY | Deep web research | [perplexity.ai](https://perplexity.ai) |
| NEWS_API_KEY | Breaking news scanning | [newsapi.org](https://newsapi.org) |
| OPENAI_API_KEY | AI image generation | [platform.openai.com](https://platform.openai.com) |

Bold = you really need these. The rest are optional extras.

## Three Ways to Talk to Royce

- **Text mode** — Type in a terminal (default)
- **Voice mode** — Hands-free, just talk
- **Hybrid mode** — Type or talk, your choice each time

## What's in the Box

```
royce-ai/
├── install_royce.bat       <- Double-click this to install
├── README.md               <- You're reading it
└── royce/
    ├── launcher.py         <- Main entry point
    ├── conversation.py     <- Royce's brain (orchestrates everything)
    ├── personality.py      <- How Royce talks and acts
    ├── memory.py           <- SQLite memory (never forgets)
    ├── voice.py            <- ElevenLabs / Polly / gTTS + mic input
    ├── mood.py             <- Mood detection from text
    ├── music.py            <- YouTube music playback
    ├── research.py         <- Deep web research (Perplexity API)
    ├── horoscope.py        <- Daily horoscope reader
    ├── news.py             <- Breaking news scanner
    ├── images.py           <- AI image generation (DALL-E / SD)
    ├── config.py           <- All settings
    ├── royce_identity.json <- Personality definition
    ├── requirements.txt    <- Python dependencies
    └── .env.example        <- API key template
```

## Requirements

- Windows 11
- Python 3.9+ ([download here](https://www.python.org/downloads/))
- Internet connection
- A microphone (for voice/hybrid mode)

The installer handles everything else automatically.
