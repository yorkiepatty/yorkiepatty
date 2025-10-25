# In Python script or interactive session
import asyncio
import websockets
import json

async def send_command():
    async with websockets.connect("ws://localhost:8000/ws/everett/derek") as ws:
        # Ask me to think deeply
        await ws.send(json.dumps({
            "command": "think",
            "about": "How to scale AlphaVox to 10,000 users while keeping it accessible"
        }))
        
        response = await ws.recv()
        print(json.loads(response)["content"])

asyncio.run(send_command())
