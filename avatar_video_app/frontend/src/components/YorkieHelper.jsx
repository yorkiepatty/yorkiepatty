import React, { useState, useEffect, useRef } from 'react'

// Path to your Yorkie image - place your yorkie.png in the public folder
// or update this path to where your image is located
const YORKIE_IMAGE = '/yorkie.png'  // Place your yorkie pic in frontend/public/yorkie.png
const YORKIE_TALKING_VIDEO = '/yorkie-talking.mp4'  // 5-second Hedra video of Yorkie talking

// Yorkie messages for each step
const YORKIE_MESSAGES = {
  1: [
    "Woof! Let's create your avatar! Describe who you want to be!",
    "Pick a style that matches your vibe! I love the cartoon one!",
    "Be creative with your description - the more detail, the better!"
  ],
  2: [
    "Now let's add your voice! Bark bark!",
    "You can record, upload, or type what you want to say!",
    "Try the voice effects - the chipmunk one is fun like me!"
  ],
  3: [
    "Almost there! Your video is being created!",
    "This is so exciting! I can't wait to see it!",
    "Woof woof! You're doing great!"
  ],
  complete: [
    "Your video is ready! You did it!",
    "That looks amazing! Time to share it!",
    "Great job! Want to make another one?"
  ],
  idle: [
    "*wags tail happily*",
    "*tilts head curiously*",
    "*panting excitedly*",
    "Click on me for tips!",
    "I'm here to help! Woof!"
  ]
}

function YorkieHelper({ currentStep, isGenerating, videoReady }) {
  const [isOpen, setIsOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [mouthOpen, setMouthOpen] = useState(false)
  const [tailWag, setTailWag] = useState(true)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const messageTimeoutRef = useRef(null)
  const mouthIntervalRef = useRef(null)
  const speechSynthRef = useRef(null)
  const videoRef = useRef(null)

  // Initialize speech synthesis
  useEffect(() => {
    if ('speechSynthesis' in window) {
      speechSynthRef.current = window.speechSynthesis
    }
    return () => {
      if (speechSynthRef.current) {
        speechSynthRef.current.cancel()
      }
    }
  }, [])

  // Get appropriate messages based on state
  const getMessages = () => {
    if (videoReady) return YORKIE_MESSAGES.complete
    if (isGenerating) return YORKIE_MESSAGES[3]
    return YORKIE_MESSAGES[currentStep] || YORKIE_MESSAGES.idle
  }

  // Speak the message using Web Speech API
  const speakMessage = (text) => {
    if (!speechSynthRef.current || !voiceEnabled) return

    // Cancel any ongoing speech
    speechSynthRef.current.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 1.2  // Slightly faster, more energetic
    utterance.pitch = 1.8  // Much higher pitch for cute dog voice
    utterance.volume = 0.8

    // Try to find a female voice - be more specific
    const voices = speechSynthRef.current.getVoices()
    // Look for specific female voices first
    let selectedVoice = voices.find(v =>
      v.name.includes('Zira') ||  // Windows female
      v.name.includes('Samantha') ||  // Mac female
      v.name.includes('Victoria') ||  // Mac female
      v.name.includes('Karen') ||  // Australian female
      v.name.includes('Moira') ||  // Irish female
      v.name.includes('Fiona') ||  // Scottish female
      (v.name.includes('Female') && v.lang.startsWith('en'))
    )
    // Fallback to any English voice if no female found
    if (!selectedVoice) {
      selectedVoice = voices.find(v => v.lang.startsWith('en'))
    }
    if (selectedVoice) {
      utterance.voice = selectedVoice
    }

    // Sync mouth animation with speech
    utterance.onstart = () => {
      setIsSpeaking(true)
      if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
      // Faster mouth movement for more natural talking look
      mouthIntervalRef.current = setInterval(() => {
        setMouthOpen(prev => !prev)
      }, 80)  // 80ms for quicker mouth movement
    }

    utterance.onend = () => {
      setIsSpeaking(false)
      if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
      setMouthOpen(false)
    }

    utterance.onerror = () => {
      setIsSpeaking(false)
      if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
      setMouthOpen(false)
    }

    speechSynthRef.current.speak(utterance)
  }

  // Show new message
  const showMessage = (customMessage = null) => {
    const messages = getMessages()
    const newMessage = customMessage || messages[Math.floor(Math.random() * messages.length)]
    setMessage(newMessage)
    setIsOpen(true)

    // Speak the message
    speakMessage(newMessage)

    // If voice not available, still animate
    if (!speechSynthRef.current || !voiceEnabled) {
      setIsSpeaking(true)
      if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
      mouthIntervalRef.current = setInterval(() => {
        setMouthOpen(prev => !prev)
      }, 100)

      if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current)
      messageTimeoutRef.current = setTimeout(() => {
        setIsSpeaking(false)
        if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
        setMouthOpen(false)
      }, newMessage.length * 50)
    }
  }

  // Toggle voice on/off
  const toggleVoice = (e) => {
    e.stopPropagation()
    setVoiceEnabled(prev => !prev)
    if (speechSynthRef.current) {
      speechSynthRef.current.cancel()
    }
  }

  // Auto-show message on step change
  useEffect(() => {
    const timer = setTimeout(() => showMessage(), 500)
    return () => clearTimeout(timer)
  }, [currentStep, isGenerating, videoReady])

  // Tail wagging animation
  useEffect(() => {
    const interval = setInterval(() => {
      setTailWag(prev => !prev)
    }, 300)
    return () => clearInterval(interval)
  }, [])

  // Play/pause talking video based on speaking state
  useEffect(() => {
    if (videoRef.current) {
      if (isSpeaking) {
        videoRef.current.play().catch(err => console.log('Video play failed:', err))
      } else {
        videoRef.current.pause()
        videoRef.current.currentTime = 0  // Reset to start
      }
    }
  }, [isSpeaking])

  // Cleanup
  useEffect(() => {
    return () => {
      if (messageTimeoutRef.current) clearTimeout(messageTimeoutRef.current)
      if (mouthIntervalRef.current) clearInterval(mouthIntervalRef.current)
    }
  }, [])

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Speech Bubble */}
      {isOpen && message && (
        <div
          className="absolute bottom-full right-0 mb-2 max-w-xs animate-fade-in"
          onClick={() => showMessage()}
        >
          <div className="bg-white rounded-2xl rounded-br-none p-4 shadow-xl relative">
            <p className="text-gray-800 text-sm font-medium">{message}</p>
            {/* Speech bubble tail */}
            <div className="absolute -bottom-2 right-4 w-4 h-4 bg-white transform rotate-45" />
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation()
              setIsOpen(false)
            }}
            className="absolute -top-2 -right-2 w-6 h-6 bg-gray-200 rounded-full text-gray-600 hover:bg-gray-300 flex items-center justify-center text-xs"
          >
            ×
          </button>
        </div>
      )}

      {/* Yorkie Avatar */}
      <button
        onClick={() => {
          if (isOpen) {
            showMessage()
          } else {
            showMessage()
          }
        }}
        className={`w-40 h-40 rounded-full shadow-xl hover:scale-110 transition-transform relative overflow-hidden ${
          isSpeaking ? 'ring-4 ring-primary-400 ring-opacity-50' : ''
        }`}
        title="Click for help!"
      >
        {/* Yorkie - show video when speaking, static image when not */}
        <div className="relative w-full h-full">
          {/* Talking video - only visible when speaking */}
          {isSpeaking && (
            <video
              ref={videoRef}
              src={YORKIE_TALKING_VIDEO}
              className="w-full h-full object-cover rounded-full"
              style={{ imageRendering: 'crisp-edges' }}
              loop
              muted
              playsInline
            />
          )}

          {/* Static image - visible when NOT speaking */}
          {!isSpeaking && (
            <img
              src={YORKIE_IMAGE}
              alt="Yorkie Helper"
              className="w-full h-full object-cover rounded-full"
            />
          )}
        </div>

        {/* Fallback avatar if no image */}
        <div
          className="w-full h-full bg-gradient-to-br from-amber-300 to-amber-500 rounded-full items-center justify-center text-4xl hidden"
        >
          🐕
        </div>

        {/* Speaking animation - pulsing ring effect */}
        {isSpeaking && (
          <div className="absolute inset-0 rounded-full border-4 border-primary-400 animate-ping opacity-30" />
        )}

        {/* Wag indicator */}
        <div
          className={`absolute -right-1 bottom-3 text-lg transform origin-left transition-transform ${
            tailWag ? 'rotate-12' : '-rotate-12'
          }`}
        >
          💕
        </div>

        {/* Speaking indicator */}
        {isSpeaking && (
          <div className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center shadow-lg">
            <div className="w-2 h-2 bg-white rounded-full animate-ping" />
          </div>
        )}

        {/* Mute/Unmute button */}
        <button
          onClick={toggleVoice}
          className="absolute -bottom-1 -left-1 w-6 h-6 bg-gray-800 hover:bg-gray-700 rounded-full flex items-center justify-center shadow-lg text-xs"
          title={voiceEnabled ? 'Mute Yorkie' : 'Unmute Yorkie'}
        >
          {voiceEnabled ? '🔊' : '🔇'}
        </button>
      </button>

      {/* Navigation hints */}
      <div className="absolute bottom-full right-0 mb-1 flex gap-1">
        {[1, 2, 3].map((step) => (
          <div
            key={step}
            className={`w-2 h-2 rounded-full transition-colors ${
              currentStep >= step
                ? 'bg-primary-400'
                : 'bg-white/30'
            }`}
            title={`Step ${step}`}
          />
        ))}
      </div>
    </div>
  )
}

export default YorkieHelper
