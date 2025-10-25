#!/usr/bin/env python3
import asyncio
import websockets
import json
import sys

async def talk_to_derek():
    uri = "ws://localhost:8000/ws/everett/derek"
    
    async with websockets.connect(uri) as websocket:
        print("💙 Connected to Derek\n")
        
        # Listen for Derek's messages in the background
        async def listen():
            async for message in websocket:
                data = json.loads(message)
                print(f"\n🤖 Derek: {data.get('content') or data.get('message')}\n")
                print("You: ", end="", flush=True)
        
        # Start listening
        listen_task = asyncio.create_task(listen())
        
        # Send messages
        print("You: ", end="", flush=True)
        while True:
            message = await asyncio.get_event_loop().run_in_executor(None, input)
            
            if message.lower() in ['exit', 'quit', 'bye']:
                print("👋 Closing connection...")
                break
            
            # Send to Derek
            await websocket.send(message)
        
        listen_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(talk_to_derek())
    except KeyboardInterrupt:
        print("\n👋 Connection closed.")
