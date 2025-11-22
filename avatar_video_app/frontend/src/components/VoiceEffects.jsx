import React, { useState, useEffect } from 'react'
import axios from 'axios'

const VOICE_EFFECTS = [
  { id: 'normal', name: 'Normal', icon: '🎤', description: 'Original voice' },
  { id: 'deep', name: 'Deep', icon: '🔊', description: 'Lower pitch' },
  { id: 'high', name: 'High', icon: '🔔', description: 'Higher pitch' },
  { id: 'robot', name: 'Robot', icon: '🤖', description: 'Robotic effect' },
  { id: 'echo', name: 'Echo', icon: '🔁', description: 'Echo/reverb' },
  { id: 'whisper', name: 'Whisper', icon: '🤫', description: 'Soft whisper' },
  { id: 'chipmunk', name: 'Chipmunk', icon: '🐿️', description: 'Very high pitch' },
  { id: 'villain', name: 'Villain', icon: '😈', description: 'Deep & ominous' },
  { id: 'announcer', name: 'Announcer', icon: '📢', description: 'Radio voice' },
  { id: 'ethereal', name: 'Ethereal', icon: '✨', description: 'Dreamy effect' }
]

function VoiceEffects({
  selectedEffect,
  setSelectedEffect,
  audioSource,
  processedAudio,
  setProcessedAudio,
  setError
}) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)

  // Preview effect when selection changes and we have audio
  useEffect(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    }
  }, [selectedEffect])

  // Process audio with selected effect
  const applyEffect = async () => {
    if (!audioSource) {
      setError('Please record or upload audio first')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('audio', audioSource)
      formData.append('effect', selectedEffect)

      const response = await axios.post('/api/voice/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      if (response.data.success) {
        const audioB64 = response.data.audio_base64
        const audioBlob = base64ToBlob(audioB64, 'audio/wav')
        setProcessedAudio({
          blob: audioBlob,
          path: response.data.audio_path,
          duration: response.data.duration,
          effect: response.data.effect_applied
        })

        const url = URL.createObjectURL(audioBlob)
        setPreviewUrl(url)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process audio')
    } finally {
      setIsProcessing(false)
    }
  }

  // Helper to convert base64 to blob
  const base64ToBlob = (base64, mimeType) => {
    const byteString = atob(base64)
    const ab = new ArrayBuffer(byteString.length)
    const ia = new Uint8Array(ab)
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i)
    }
    return new Blob([ab], { type: mimeType })
  }

  return (
    <div className="card">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-accent-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        Voice Effects
      </h3>

      {/* Effect Grid */}
      <div className="grid grid-cols-5 gap-2 mb-4">
        {VOICE_EFFECTS.map((effect) => (
          <button
            key={effect.id}
            onClick={() => setSelectedEffect(effect.id)}
            className={`p-3 rounded-xl text-center transition-all group relative ${
              selectedEffect === effect.id
                ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white ring-2 ring-white/20'
                : 'bg-white/5 text-white/70 hover:bg-white/10'
            }`}
            title={effect.description}
          >
            <div className="text-xl mb-1">{effect.icon}</div>
            <div className="text-xs font-medium truncate">{effect.name}</div>

            {/* Tooltip */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
              {effect.description}
            </div>
          </button>
        ))}
      </div>

      {/* Apply Button */}
      {audioSource && selectedEffect !== 'normal' && (
        <button
          onClick={applyEffect}
          disabled={isProcessing}
          className="btn-secondary w-full mb-4"
        >
          {isProcessing ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Applying Effect...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Preview with {VOICE_EFFECTS.find(e => e.id === selectedEffect)?.name} Effect
            </span>
          )}
        </button>
      )}

      {/* Preview Audio */}
      {previewUrl && (
        <div className="p-3 bg-accent-500/10 border border-accent-500/30 rounded-xl">
          <div className="flex items-center gap-2 mb-2 text-accent-400 text-sm">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            Effect Applied: {VOICE_EFFECTS.find(e => e.id === selectedEffect)?.name}
          </div>
          <audio controls src={previewUrl} className="w-full" />
        </div>
      )}

      {/* Info */}
      {!audioSource && (
        <div className="text-white/40 text-sm text-center py-4">
          Record or upload audio to preview voice effects
        </div>
      )}
    </div>
  )
}

export default VoiceEffects
