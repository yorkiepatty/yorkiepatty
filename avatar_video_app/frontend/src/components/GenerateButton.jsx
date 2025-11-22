import React from 'react'
import axios from 'axios'

function GenerateButton({
  canGenerate,
  isGenerating,
  setIsGenerating,
  avatarDescription,
  avatarStyle,
  avatarPath,
  voiceMode,
  recordedAudio,
  uploadedAudio,
  ttsText,
  voiceEffect,
  processedAudio,
  setGeneratedVideo,
  setVideoJobId,
  setVideoStatus,
  setCurrentStep,
  setError
}) {
  const generateVideo = async () => {
    setIsGenerating(true)
    setError(null)
    setCurrentStep(3)

    try {
      const formData = new FormData()
      formData.append('avatar_description', avatarDescription)
      formData.append('avatar_style', avatarStyle)
      formData.append('voice_effect', voiceEffect)

      // Add audio source
      if (voiceMode === 'tts') {
        formData.append('text', ttsText)
      } else if (processedAudio?.blob) {
        formData.append('audio', processedAudio.blob, 'voice.wav')
      } else if (recordedAudio) {
        formData.append('audio', recordedAudio, 'recording.webm')
      } else if (uploadedAudio) {
        formData.append('audio', uploadedAudio, uploadedAudio.name)
      }

      const response = await axios.post('/api/generate-full', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000 // 5 minute timeout
      })

      if (response.data.success) {
        const videoData = response.data.video

        if (videoData.status === 'completed') {
          // Video is ready
          setGeneratedVideo({
            path: videoData.video_path,
            base64: videoData.video_base64,
            duration: response.data.voice?.duration
          })
          setVideoStatus('completed')
        } else if (videoData.status === 'processing') {
          // Video is being processed
          setVideoJobId(videoData.job_id)
          setVideoStatus('processing')
        }
      } else {
        setError('Video generation failed')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate video')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="text-center">
      <button
        onClick={generateVideo}
        disabled={!canGenerate || isGenerating}
        className={`btn-primary text-lg py-4 px-12 ${
          !canGenerate ? 'opacity-50 cursor-not-allowed' : ''
        }`}
      >
        {isGenerating ? (
          <span className="flex items-center gap-3">
            <svg className="animate-spin w-6 h-6" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Generating Your Video...
          </span>
        ) : (
          <span className="flex items-center gap-3">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Generate Talking Avatar Video
          </span>
        )}
      </button>

      {!canGenerate && (
        <p className="mt-3 text-white/50 text-sm">
          {!avatarPath
            ? 'Create an avatar first'
            : 'Add voice (record, upload, or enter text)'
          }
        </p>
      )}

      {isGenerating && (
        <div className="mt-4 text-white/60 text-sm">
          <p>This may take a few minutes depending on video length...</p>
          <div className="flex justify-center gap-2 mt-2">
            {['Creating avatar', 'Processing voice', 'Generating video'].map((step, i) => (
              <span key={step} className="flex items-center gap-1">
                <span className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: `${i * 0.3}s` }} />
                <span>{step}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default GenerateButton
