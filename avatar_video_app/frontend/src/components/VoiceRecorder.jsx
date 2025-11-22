import React, { useState, useRef, useEffect } from 'react'

const MAX_DURATION = 180 // 3 minutes in seconds

function VoiceRecorder({
  mode,
  setMode,
  recordedAudio,
  setRecordedAudio,
  ttsText,
  setTtsText,
  uploadedAudio,
  setUploadedAudio,
  audioDuration,
  setAudioDuration,
  setError
}) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioUrl, setAudioUrl] = useState(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const fileInputRef = useRef(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setAudioUrl(url)
        setRecordedAudio(blob)
        setAudioDuration(recordingTime)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= MAX_DURATION) {
            stopRecording()
            return prev
          }
          return prev + 1
        })
      }, 1000)

    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.')
    }
  }

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }

  // Format time display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Check file type
    if (!file.type.startsWith('audio/')) {
      setError('Please upload an audio file (MP3, WAV, etc.)')
      return
    }

    // Check file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('File too large. Maximum size is 50MB.')
      return
    }

    setUploadedAudio(file)

    // Get duration
    const url = URL.createObjectURL(file)
    const audio = new Audio(url)
    audio.addEventListener('loadedmetadata', () => {
      if (audio.duration > MAX_DURATION) {
        setError(`Audio too long. Maximum duration is ${MAX_DURATION / 60} minutes.`)
        setUploadedAudio(null)
      } else {
        setAudioDuration(audio.duration)
        setAudioUrl(url)
      }
    })
  }

  // Clear current audio
  const clearAudio = () => {
    setRecordedAudio(null)
    setUploadedAudio(null)
    setAudioUrl(null)
    setAudioDuration(0)
    setRecordingTime(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span className="w-8 h-8 bg-primary-500/20 rounded-lg flex items-center justify-center text-primary-400">
          2
        </span>
        Add Voice
      </h2>

      {/* Mode Selection */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'record', label: 'Record', icon: '🎙️' },
          { id: 'upload', label: 'Upload', icon: '📁' },
          { id: 'tts', label: 'Text to Speech', icon: '💬' }
        ].map((m) => (
          <button
            key={m.id}
            onClick={() => {
              setMode(m.id)
              clearAudio()
            }}
            className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all ${
              mode === m.id
                ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white'
                : 'bg-white/5 text-white/70 hover:bg-white/10'
            }`}
          >
            <span className="text-lg mr-2">{m.icon}</span>
            {m.label}
          </button>
        ))}
      </div>

      {/* Record Mode */}
      {mode === 'record' && (
        <div className="text-center">
          {/* Recording Visualizer */}
          <div className="h-24 flex items-center justify-center gap-1 mb-4">
            {isRecording ? (
              Array.from({ length: 20 }).map((_, i) => (
                <div
                  key={i}
                  className="w-2 bg-gradient-to-t from-primary-500 to-accent-500 rounded-full waveform-bar"
                  style={{
                    animationDelay: `${i * 0.05}s`,
                    height: `${20 + Math.random() * 60}%`
                  }}
                />
              ))
            ) : audioUrl ? (
              <audio controls src={audioUrl} className="w-full" />
            ) : (
              <div className="text-white/40">
                <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <p className="mt-2 text-sm">Click to start recording</p>
              </div>
            )}
          </div>

          {/* Timer */}
          <div className="text-3xl font-mono text-white mb-4">
            {formatTime(recordingTime)} / {formatTime(MAX_DURATION)}
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-white/10 rounded-full overflow-hidden mb-6">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500 transition-all"
              style={{ width: `${(recordingTime / MAX_DURATION) * 100}%` }}
            />
          </div>

          {/* Record Button */}
          <div className="flex justify-center gap-4">
            {!isRecording && !audioUrl && (
              <button
                onClick={startRecording}
                className="w-20 h-20 bg-gradient-to-r from-red-500 to-pink-500 rounded-full flex items-center justify-center hover:scale-105 transition-transform shadow-lg"
              >
                <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 15c1.66 0 3-1.34 3-3V6c0-1.66-1.34-3-3-3S9 4.34 9 6v6c0 1.66 1.34 3 3 3z" />
                  <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                </svg>
              </button>
            )}

            {isRecording && (
              <button
                onClick={stopRecording}
                className="px-8 py-4 bg-gradient-to-r from-red-600 to-red-500 rounded-xl flex items-center justify-center gap-3 hover:scale-105 transition-transform shadow-lg animate-pulse"
              >
                <div className="w-6 h-6 bg-white rounded" />
                <span className="text-white font-bold text-xl">STOP RECORDING</span>
              </button>
            )}

            {audioUrl && !isRecording && (
              <>
                <button
                  onClick={clearAudio}
                  className="btn-secondary"
                >
                  Record Again
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Upload Mode */}
      {mode === 'upload' && (
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {!audioUrl ? (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full h-40 border-2 border-dashed border-white/20 rounded-xl hover:border-primary-500/50 hover:bg-white/5 transition-all flex flex-col items-center justify-center text-white/50"
            >
              <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span>Click to upload audio file</span>
              <span className="text-xs mt-1">MP3, WAV, M4A (max 50MB)</span>
            </button>
          ) : (
            <div className="space-y-4">
              <audio controls src={audioUrl} className="w-full" />
              <div className="flex items-center justify-between text-sm text-white/60">
                <span>Duration: {formatTime(Math.floor(audioDuration))}</span>
                <button onClick={clearAudio} className="text-red-400 hover:text-red-300">
                  Remove
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TTS Mode */}
      {mode === 'tts' && (
        <div>
          <label className="label">Enter text to speak (max 3 minutes of speech)</label>
          <textarea
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            placeholder="Type what you want your avatar to say... The text will be converted to speech with your selected voice effect."
            className="input-field h-40 resize-none"
            maxLength={5000}
          />
          <div className="flex justify-between text-white/40 text-xs mt-1">
            <span>Estimated duration: ~{Math.ceil(ttsText.split(' ').length / 150)} min</span>
            <span>{ttsText.length}/5000</span>
          </div>
        </div>
      )}

      {/* Duration Warning */}
      {audioDuration > 0 && audioDuration <= MAX_DURATION && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-xl text-green-400 text-sm flex items-center gap-2">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          Audio ready: {formatTime(Math.floor(audioDuration))}
        </div>
      )}
    </div>
  )
}

export default VoiceRecorder
