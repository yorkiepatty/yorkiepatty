// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/derek');

ws.onopen = () => {
    console.log('✅ Connected to Sonny API');
    
    // Send TTS request
    ws.send(JSON.stringify({
        command: 'tts',
        payload: { text: 'Hello from AlphaVox!' }
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('📨 Received:', response);
    
    if (response.type === 'tts_response') {
        // Play audio
        const audio = new Audio(response.data.audio);
        audio.play();
    }
};

ws.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
};

ws.onclose = () => {
    console.log('❌ Disconnected from Sonny API');
};
