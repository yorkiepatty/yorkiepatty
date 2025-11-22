import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const AVATAR_STYLES = [
  { id: 'realistic', name: 'Realistic', icon: '📷' },
  { id: 'anime', name: 'Anime', icon: '🎌' },
  { id: 'cartoon', name: 'Cartoon', icon: '🎨' },
  { id: '3d_render', name: '3D Render', icon: '🎮' },
  { id: 'artistic', name: 'Artistic', icon: '🖼️' },
  { id: 'pixel_art', name: 'Pixel Art', icon: '👾' },
  { id: 'watercolor', name: 'Watercolor', icon: '💧' },
  { id: 'oil_painting', name: 'Oil Painting', icon: '🎭' }
]

function AvatarCreator({
  description,
  setDescription,
  style,
  setStyle,
  preview,
  setPreview,
  avatarPath,
  setAvatarPath,
  onComplete,
  setError
}) {
  const [isGenerating, setIsGenerating] = useState(false)
  const [mode, setMode] = useState('generate') // 'generate' or 'upload'
  const fileInputRef = useRef(null)

  // Handle image upload
  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Check file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (PNG, JPG, etc.)')
      return
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Image too large. Maximum size is 10MB.')
      return
    }

    setIsGenerating(true)
    setError(null)

    try {
      // Upload to server
      const formData = new FormData()
      formData.append('image', file)

      const response = await axios.post('/api/avatar/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      if (response.data.success) {
        setPreview(`data:image/png;base64,${response.data.image_base64}`)
        setAvatarPath(response.data.image_path)
        setDescription('Uploaded image')
        onComplete()
      } else {
        setError('Failed to upload image')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload image')
    } finally {
      setIsGenerating(false)
    }
  }

  const generateAvatar = async () => {
    if (!description.trim()) {
      setError('Please enter a description for your avatar')
      return
    }

    setIsGenerating(true)
    setError(null)

    try {
      const response = await axios.post('/api/avatar/generate', {
        description: description.trim(),
        style: style,
        use_cache: true
      })

      if (response.data.success) {
        setPreview(`data:image/png;base64,${response.data.image_base64}`)
        setAvatarPath(response.data.image_path)
        onComplete()
      } else {
        setError('Failed to generate avatar')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate avatar')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="card h-full">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span className="w-8 h-8 bg-primary-500/20 rounded-lg flex items-center justify-center text-primary-400">
          1
        </span>
        Create Your Avatar
      </h2>

      {/* Mode Selection - Generate or Upload */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setMode('generate')}
          className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all ${
            mode === 'generate'
              ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white'
              : 'bg-white/5 text-white/70 hover:bg-white/10'
          }`}
        >
          ✨ Generate New
        </button>
        <button
          onClick={() => setMode('upload')}
          className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all ${
            mode === 'upload'
              ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white'
              : 'bg-white/5 text-white/70 hover:bg-white/10'
          }`}
        >
          📁 Upload Image
        </button>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
      />

      {mode === 'generate' ? (
        <>
          {/* Description Input */}
          <div className="mb-4">
            <label className="label">Describe Your Avatar</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g., A friendly young woman with long brown hair, green eyes, wearing a blue blazer, professional look, warm smile..."
              className="input-field h-32 resize-none"
              maxLength={500}
            />
            <div className="text-right text-white/40 text-xs mt-1">
              {description.length}/500
            </div>
          </div>

          {/* Style Selection */}
          <div className="mb-6">
            <label className="label">Choose Style</label>
            <div className="grid grid-cols-4 gap-2">
              {AVATAR_STYLES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setStyle(s.id)}
                  className={`p-3 rounded-xl text-center transition-all ${
                    style === s.id
                      ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white ring-2 ring-white/20'
                      : 'bg-white/5 text-white/70 hover:bg-white/10'
                  }`}
                >
                  <div className="text-2xl mb-1">{s.icon}</div>
                  <div className="text-xs font-medium">{s.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={generateAvatar}
            disabled={isGenerating || !description.trim()}
            className="btn-primary w-full mb-6"
          >
            {isGenerating ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generating Avatar...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Generate Avatar
              </span>
            )}
          </button>
        </>
      ) : (
        <>
          {/* Upload Section */}
          <div className="mb-6">
            <p className="text-white/60 text-sm mb-4">
              Upload any image to use as your avatar. Previously generated avatars are saved in the <code className="bg-white/10 px-1 rounded">avatar_outputs</code> folder.
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isGenerating}
              className="btn-primary w-full"
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Uploading...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Choose Image File
                </span>
              )}
            </button>
          </div>
        </>
      )}

      {/* Preview */}
      <div className="relative aspect-square rounded-xl overflow-hidden bg-white/5 border-2 border-dashed border-white/20">
        {preview ? (
          <img
            src={preview}
            alt="Generated Avatar"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/40">
            <svg className="w-16 h-16 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <span className="text-sm">Your avatar will appear here</span>
          </div>
        )}
        {isGenerating && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-center">
              <svg className="animate-spin w-12 h-12 text-primary-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span className="text-white/70 text-sm">Creating your avatar...</span>
            </div>
          </div>
        )}
      </div>

      {avatarPath && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-xl text-green-400 text-sm flex items-center gap-2">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          Avatar created successfully!
        </div>
      )}
    </div>
  )
}

export default AvatarCreator
