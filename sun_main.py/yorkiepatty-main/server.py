from flask import Flask, request, jsonify
from speech_recognition_engine import get_speech_recognition_engine

app = Flask(__name__)
engine = get_speech_recognition_engine(simulate=False)


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    # Feed directly to recognizer
    try:
        text, confidence, meta = engine.recognize_from_bytes(audio_bytes)
        return jsonify({"text": text, "confidence": confidence, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
