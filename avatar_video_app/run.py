#!/usr/bin/env python3
"""
Avatar Video App - Run Script
Starts the backend API server
"""
import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run Avatar Video App API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    try:
        import uvicorn
        from avatar_video_app.api import app

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  Avatar Video Creator                         ║
║              Create 3-minute Talking Avatars                  ║
╠══════════════════════════════════════════════════════════════╣
║  API Server: http://{args.host}:{args.port}                          ║
║  API Docs:   http://{args.host}:{args.port}/docs                     ║
║                                                               ║
║  Frontend:   cd frontend && npm install && npm run dev       ║
║              Then visit http://localhost:3000                 ║
╚══════════════════════════════════════════════════════════════╝
        """)

        uvicorn.run(
            "avatar_video_app.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install uvicorn fastapi")
        sys.exit(1)


if __name__ == "__main__":
    main()
