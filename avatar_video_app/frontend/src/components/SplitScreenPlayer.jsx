import React, { useState, useEffect, useRef } from 'react'

function SplitScreenPlayer({ conversation, onClose }) {
  const { character1, character2, script, videos } = conversation
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const videoRef1 = useRef(null)
  const videoRef2 = useRef(null)

  const currentLine = script[currentLineIndex]
  const currentVideo = videos?.[currentLineIndex]

  useEffect(() => {
    if (isPlaying && currentVideo) {
      playCurrentVideo()
    }
  }, [currentLineIndex, isPlaying])

  const playCurrentVideo = () => {
    const videoRef = currentLine.speaker === 1 ? videoRef1 : videoRef2
    if (videoRef.current && currentVideo) {
      videoRef.current.src = currentVideo.url || `data:video/mp4;base64,${currentVideo.base64}`
      videoRef.current.play()
    }
  }

  const handleVideoEnd = () => {
    if (currentLineIndex < script.length - 1) {
      setCurrentLineIndex(currentLineIndex + 1)
    } else {
      setIsPlaying(false)
      setCurrentLineIndex(0)
    }
  }

  const handlePlay = () => {
    setIsPlaying(true)
    setCurrentLineIndex(0)
  }

  const handlePause = () => {
    setIsPlaying(false)
    const videoRef = currentLine.speaker === 1 ? videoRef1 : videoRef2
    if (videoRef.current) {
      videoRef.current.pause()
    }
  }

  const handleNext = () => {
    if (currentLineIndex < script.length - 1) {
      setCurrentLineIndex(currentLineIndex + 1)
    }
  }

  const handlePrevious = () => {
    if (currentLineIndex > 0) {
      setCurrentLineIndex(currentLineIndex - 1)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <button
          onClick={onClose}
          className="mb-6 text-white/80 hover:text-white flex items-center gap-2"
        >
          ← Back
        </button>

        <h1 className="text-3xl font-bold text-white mb-8">Split Screen Conversation</h1>

        {/* Split Screen Display */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Character 1 Side */}
          <div className={`relative bg-white/10 backdrop-blur-lg border-4 rounded-2xl overflow-hidden ${
            currentLine?.speaker === 1 ? 'border-primary-400 shadow-lg shadow-primary-500/50' : 'border-white/20'
          }`}>
            <div className="aspect-[9/16] bg-black relative">
              <video
                ref={videoRef1}
                className="w-full h-full object-cover"
                onEnded={handleVideoEnd}
                playsInline
              />
              {currentLine?.speaker !== 1 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  {character1.avatarImage ? (
                    <img
                      src={character1.avatarImage}
                      alt={character1.name}
                      className="w-full h-full object-cover opacity-50"
                    />
                  ) : (
                    <div className="text-6xl text-white/30">👤</div>
                  )}
                </div>
              )}
            </div>
            <div className="p-4 bg-gradient-to-r from-primary-600 to-primary-500">
              <h3 className="text-xl font-bold text-white">{character1.name}</h3>
              <p className="text-white/80 text-sm capitalize">{character1.personality}</p>
            </div>
          </div>

          {/* Character 2 Side */}
          <div className={`relative bg-white/10 backdrop-blur-lg border-4 rounded-2xl overflow-hidden ${
            currentLine?.speaker === 2 ? 'border-accent-400 shadow-lg shadow-accent-500/50' : 'border-white/20'
          }`}>
            <div className="aspect-[9/16] bg-black relative">
              <video
                ref={videoRef2}
                className="w-full h-full object-cover"
                onEnded={handleVideoEnd}
                playsInline
              />
              {currentLine?.speaker !== 2 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  {character2.avatarImage ? (
                    <img
                      src={character2.avatarImage}
                      alt={character2.name}
                      className="w-full h-full object-cover opacity-50"
                    />
                  ) : (
                    <div className="text-6xl text-white/30">👤</div>
                  )}
                </div>
              )}
            </div>
            <div className="p-4 bg-gradient-to-r from-accent-600 to-accent-500">
              <h3 className="text-xl font-bold text-white">{character2.name}</h3>
              <p className="text-white/80 text-sm capitalize">{character2.personality}</p>
            </div>
          </div>
        </div>

        {/* Current Dialogue Display */}
        {currentLine && (
          <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-6 mb-6">
            <div className="flex items-center gap-4 mb-3">
              <div className={`w-3 h-3 rounded-full ${
                currentLine.speaker === 1 ? 'bg-primary-400' : 'bg-accent-400'
              }`} />
              <span className="font-bold text-white">
                {currentLine.speaker === 1 ? character1.name : character2.name}
              </span>
              <span className="text-white/50 text-sm">
                Line {currentLineIndex + 1} of {script.length}
              </span>
            </div>
            <p className="text-white text-lg">{currentLine.text}</p>
          </div>
        )}

        {/* Controls */}
        <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-6">
          <div className="flex items-center justify-center gap-4 mb-4">
            <button
              onClick={handlePrevious}
              disabled={currentLineIndex === 0}
              className="p-3 bg-white/10 hover:bg-white/20 disabled:bg-white/5 disabled:opacity-50 rounded-full text-white"
            >
              ⏮
            </button>

            {!isPlaying ? (
              <button
                onClick={handlePlay}
                disabled={!videos || videos.length === 0}
                className="p-4 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-600 disabled:to-gray-600 disabled:opacity-50 rounded-full text-white text-2xl"
              >
                ▶
              </button>
            ) : (
              <button
                onClick={handlePause}
                className="p-4 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 rounded-full text-white text-2xl"
              >
                ⏸
              </button>
            )}

            <button
              onClick={handleNext}
              disabled={currentLineIndex === script.length - 1}
              className="p-3 bg-white/10 hover:bg-white/20 disabled:bg-white/5 disabled:opacity-50 rounded-full text-white"
            >
              ⏭
            </button>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-white/10 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-primary-400 to-accent-400 h-2 rounded-full transition-all"
              style={{ width: `${((currentLineIndex + 1) / script.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Script Preview */}
        <div className="mt-6 bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-6">
          <h3 className="text-xl font-bold text-white mb-4">Full Script</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {script.map((line, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg cursor-pointer transition-all ${
                  index === currentLineIndex
                    ? 'bg-white/20 border-2 border-white/40'
                    : 'bg-white/5 border border-white/10 hover:bg-white/10'
                }`}
                onClick={() => setCurrentLineIndex(index)}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-2 h-2 rounded-full ${
                    line.speaker === 1 ? 'bg-primary-400' : 'bg-accent-400'
                  }`} />
                  <span className="font-bold text-white text-sm">
                    {line.speaker === 1 ? character1.name : character2.name}
                  </span>
                </div>
                <p className="text-white/80 text-sm pl-4">{line.text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SplitScreenPlayer
