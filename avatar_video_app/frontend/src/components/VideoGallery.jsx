import React, { useState, useEffect } from 'react'
import axios from 'axios'

function VideoGallery() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedVideo, setSelectedVideo] = useState(null)

  // Fetch videos on mount
  useEffect(() => {
    fetchVideos()
  }, [])

  const fetchVideos = async () => {
    try {
      const response = await axios.get('/api/video/list')
      setVideos(response.data.videos || [])
    } catch (err) {
      console.error('Failed to fetch videos:', err)
    } finally {
      setLoading(false)
    }
  }

  // Format file size
  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  // Format date
  const formatDate = (isoDate) => {
    const date = new Date(isoDate)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <svg className="animate-spin w-6 h-6 text-primary-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="ml-2 text-white/60">Loading videos...</span>
        </div>
      </div>
    )
  }

  if (videos.length === 0) {
    return null // Don't show empty gallery
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          Your Videos
        </h2>
        <button
          onClick={fetchVideos}
          className="text-white/60 hover:text-white transition-colors"
          title="Refresh"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Video Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {videos.map((video) => (
          <div
            key={video.name}
            onClick={() => setSelectedVideo(video)}
            className="group cursor-pointer"
          >
            <div className="aspect-[9/16] bg-white/5 rounded-xl overflow-hidden relative hover:ring-2 hover:ring-primary-500/50 transition-all">
              {/* Video Thumbnail/Preview */}
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-black/20 to-black/60">
                <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </div>
              </div>

              {/* Info Overlay */}
              <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent">
                <p className="text-white text-xs truncate">{video.name}</p>
                <p className="text-white/60 text-xs">{formatSize(video.size)}</p>
              </div>
            </div>

            <p className="text-white/40 text-xs mt-1 text-center">
              {formatDate(video.created)}
            </p>
          </div>
        ))}
      </div>

      {/* Video Modal */}
      {selectedVideo && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedVideo(null)}
        >
          <div
            className="bg-gray-900 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <h3 className="font-medium text-white truncate">{selectedVideo.name}</h3>
              <button
                onClick={() => setSelectedVideo(null)}
                className="text-white/60 hover:text-white"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Video Player */}
            <div className="aspect-[9/16] max-h-[60vh] bg-black">
              <video
                src={`/outputs/${selectedVideo.name}`}
                controls
                autoPlay
                className="w-full h-full object-contain"
              />
            </div>

            {/* Actions */}
            <div className="p-4 flex justify-between items-center">
              <div className="text-white/60 text-sm">
                <p>Size: {formatSize(selectedVideo.size)}</p>
                <p>Created: {formatDate(selectedVideo.created)}</p>
              </div>
              <a
                href={`/api/video/download/${selectedVideo.name}`}
                download
                className="btn-primary py-2 px-4 text-sm"
              >
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </span>
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default VideoGallery
