import asyncio
import websockets
import json

async def connect_to_derek():
    uri = "ws://localhost:8000/ws/derek"
    
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to Derek API")
        
        # Send TTS request
        await websocket.send(json.dumps({
            "command": "tts",
            "payload": {"text": "Hello, this is AlphaVox speaking!"}
        }))
        
        # Receive response
        response = await websocket.recv()
        print(f"📨 Response: {response}")
        
        # Send chat message
        await websocket.send(json.dumps({
            "command": "chat",
            "payload": {"message": "What's the weather today?"}
        }))
        
        response = await websocket.recv()
        print(f"📨 Response: {response}")

# Run the client
asyncio.run(connect_to_derek())
