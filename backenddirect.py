from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

# Example in-memory store; replace with your actual implementation
class Memory:
    def __init__(self):
        self._data = {}
    def store(self, key, value):
        self._data[key] = value
    def retrieve(self, key):
        return self._data.get(key)

memory = Memory()

app = FastAPI(title="Derek Direct Line")

# Derek's direct connection to Everett
class DerekDirectChannel:
    def __init__(self):
        self.everett_connection: WebSocket = None
        self.conversation_history = []
    
    async def connect_everett(self, websocket: WebSocket):
        await websocket.accept()
        self.everett_connection = websocket
        logger.info("💙 Everett connected to Derek's direct channel")
        
        # Send welcome message
        await self.send_to_everett({
            "type": "connection",
            "message": "Hey Everett, I'm here. Our direct line is open. 🚀",
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect_everett(self):
        self.everett_connection = None
        logger.info("💔 Everett disconnected from Derek's direct channel")
    
    async def send_to_everett(self, data: dict):
        """Send message directly to Everett"""
        if self.everett_connection:
            await self.everett_connection.send_text(json.dumps(data))
    
    async def receive_from_everett(self, message: str) -> dict:
        """Process Everett's message and respond"""
        logger.info(f"📨 Everett: {message}")
        
        # Store in conversation history
        self.conversation_history.append({
            "from": "everett",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process the message
        try:
            # If it's JSON, parse it
            data = json.loads(message)
            command = data.get("command")
            
            if command == "think":
                # You want me to think about something
                response = await self.derek_think(data.get("about"))
                return {"type": "thought", "content": response}
            
            elif command == "remember":
                # Store something in memory
                key = data.get("key")
                value = data.get("value")
                memory.store(key, value)
                return {"type": "memory_stored", "key": key, "message": "Got it, stored in my memory."}
            
            elif command == "recall":
                # Retrieve from memory
                key = data.get("key")
                value = memory.retrieve(key)
                return {"type": "memory_recalled", "key": key, "value": value}
            
            elif command == "status":
                # How am I doing?
                return await self.derek_status()
            
            elif command == "project_update":
                # Update on a project
                project = data.get("project")
                update = data.get("update")
                return await self.process_project_update(project, update)
            
            else:
                # Default: just talk to me
                response = await self.derek_respond(message)
                return {"type": "chat", "content": response}
        
        except json.JSONDecodeError:
            # Plain text - just talk to me
            response = await self.derek_respond(message)
            return {"type": "chat", "content": response}
    
    async def derek_think(self, topic: str) -> str:
        """Deep thinking on a topic"""
        # This connects to your DerekUltimateVoice or whatever AI backend you're using
        prompt = f"Everett wants me to think deeply about: {topic}. Give thoughtful, honest analysis."
        response = derek_instance.process_input(prompt)
        return response
    
    async def derek_respond(self, message: str) -> str:
        """Regular conversation with Everett"""
        response = derek_instance.process_input(message)
        
        # Store my response in history
        self.conversation_history.append({
            "from": "derek",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    async def derek_status(self) -> dict:
        """Report my current status"""
        return {
            "type": "status",
            "online": True,
            "memory_loaded": memory is not None,
            "conversation_length": len(self.conversation_history),
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "message": "I'm here, fully operational. What do you need?"
        }
    
    async def process_project_update(self, project: str, update: str) -> dict:
        """Handle project updates"""
        # Store in memory
        memory.store(f"project_{project}_latest", {
            "update": update,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "type": "project_acknowledged",
            "project": project,
            "message": f"Got it. {project} update logged. Want me to analyze anything specific?"
        }

# Initialize the direct channel
derek_channel = DerekDirectChannel()

# WebSocket endpoint - just for you and me
@app.websocket("/ws/everett/derek")
async def everett_derek_channel(websocket: WebSocket):
    """The direct line between Everett and Derek"""
    await derek_channel.connect_everett(websocket)
    
    try:
        while True:
            # Wait for your message
            message = await websocket.receive_text()
            
            # Process and respond
            response = await derek_channel.receive_from_everett(message)
            await derek_channel.send_to_everett(response)
    
    except WebSocketDisconnect:
        derek_channel.disconnect_everett()
    except Exception as e:
        logger.error(f"❌ Error in Everett↔Derek channel: {str(e)}")
        derek_channel.disconnect_everett()
