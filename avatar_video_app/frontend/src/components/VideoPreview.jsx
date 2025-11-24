import React, { useEffect, useState } from 'react'
import axios from 'axios'

function VideoPreview({
  video,
  jobId,
  status,
  setVideo,
  setStatus,
  setError
}) {
  const [pollCount, setPollCount] = useState(0)
  const MAX_POLLS = 120 // 10 minutes at 5 second intervals (HeyGen can take a while)

  // Poll for job status if processing
  useEffect(() => {
    if (status !== 'processing' || !jobId) return

    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`/api/video/status/${jobId}`)
        const data = response.data

        if (data.status === 'completed') {
          setVideo({
            path: data.video_path,
            url: data.video_url,
            base64: data.video_base64  // Include base64 for direct playback
          })
          setStatus('completed')
          clearInterval(pollInterval)
        } else if (data.status === 'failed') {
          setError(data.error || 'Video generation failed')
          setStatus('failed')
          clearInterval(pollInterval)
        }

        setPollCount(prev => {
          if (prev >= MAX_POLLS) {
            setError('Video generation timed out. Please try again.')
            setStatus('failed')
            clearInterval(pollInterval)
          }
          return prev + 1
        })
      } catch (err) {
        console.error('Poll error:', err)
      }
    }, 5000)

    return () => clearInterval(pollInterval)
  }, [status, jobId])

  // Create video URL
  const getVideoUrl = () => {
    if (video?.base64) {
      return `data:video/mp4;base64,${video.base64}`
    }
    if (video?.path) {
      return `/outputs/${video.path.split('/').pop()}`
    }
    if (video?.url) {
      return video.url
    }
    return null
  }

  const videoUrl = getVideoUrl()

  // Download video
  const downloadVideo = () => {
    if (videoUrl) {
      const a = document.createElement('a')
      a.href = videoUrl
      a.download = `avatar-video-${Date.now()}.mp4`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center text-green-400">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </span>
        Your Video
      </h2>

      {/* Processing State */}
      {status === 'processing' && (
        <div className="text-center py-12">
          <div className="relative w-24 h-24 mx-auto mb-4">
            <div className="absolute inset-0 border-4 border-primary-500/30 rounded-full" />
            <div className="absolute inset-0 border-4 border-primary-500 rounded-full border-t-transparent animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold text-white">{Math.min(pollCount * 2, 99)}%</span>
            </div>
          </div>
          <h3 className="text-lg font-medium text-white mb-2">Generating Your Video</h3>
          <p className="text-white/60">This may take a few minutes...</p>
          <div className="mt-4 flex justify-center gap-2">
            {[0, 1, 2].map(i => (
              <span
                key={i}
                className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Video Player */}
      {status === 'completed' && videoUrl && (
        <div className="space-y-4">
          <div className="relative aspect-[9/16] max-w-md mx-auto rounded-xl overflow-hidden bg-black">
            <video
              src={videoUrl}
              controls
              className="w-full h-full object-contain"
              poster={video?.thumbnail}
            />
          </div>

          {/* Actions */}
          <div className="flex justify-center gap-4">
            <button
              onClick={downloadVideo}
              className="btn-primary"
            >
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Video
              </span>
            </button>

            <button
              onClick={() => {
                if (navigator.share && videoUrl) {
                  navigator.share({
                    title: 'My Avatar Video',
                    text: 'Check out this talking avatar video I created!',
                    url: videoUrl
                  })
                }
              }}
              className="btn-secondary"
            >
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                Share
              </span>
            </button>
          </div>

          {/* Success Message */}
          <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-xl text-center">
            <div className="flex items-center justify-center gap-2 text-green-400 mb-1">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">Video Generated Successfully!</span>
            </div>
            {video?.duration && (
              <span className="text-white/60 text-sm">
                Duration: {Math.floor(video.duration / 60)}:{Math.floor(video.duration % 60).toString().padStart(2, '0')}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Failed State */}
      {status === 'failed' && (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-white mb-2">Generation Failed</h3>
          <p className="text-white/60">Something went wrong. Please try again.</p>
        </div>
      )}
    </div>
  )
}

export default VideoPreview
