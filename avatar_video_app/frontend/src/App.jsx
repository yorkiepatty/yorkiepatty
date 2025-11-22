import React, { useState, useCallback } from 'react'
import Header from './components/Header'
import AvatarCreator from './components/AvatarCreator'
import VoiceRecorder from './components/VoiceRecorder'
import VoiceEffects from './components/VoiceEffects'
import VideoPreview from './components/VideoPreview'
import GenerateButton from './components/GenerateButton'
import VideoGallery from './components/VideoGallery'
import YorkieHelper from './components/YorkieHelper'
import Settings from './components/Settings'

function App() {
  // State for the entire app
  const [currentStep, setCurrentStep] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState(null)

  // Avatar state
  const [avatarDescription, setAvatarDescription] = useState('')
  const [avatarStyle, setAvatarStyle] = useState('realistic')
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [avatarPath, setAvatarPath] = useState(null)

  // Voice state
  const [voiceMode, setVoiceMode] = useState('record') // 'record' | 'tts' | 'upload'
  const [recordedAudio, setRecordedAudio] = useState(null)
  const [ttsText, setTtsText] = useState('')
  const [uploadedAudio, setUploadedAudio] = useState(null)
  const [voiceEffect, setVoiceEffect] = useState('normal')
  const [processedAudio, setProcessedAudio] = useState(null)
  const [audioDuration, setAudioDuration] = useState(0)

  // Video state
  const [generatedVideo, setGeneratedVideo] = useState(null)
  const [videoJobId, setVideoJobId] = useState(null)
  const [videoStatus, setVideoStatus] = useState(null)

  // Settings state
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Reset all state
  const resetAll = useCallback(() => {
    setCurrentStep(1)
    setError(null)
    setAvatarDescription('')
    setAvatarStyle('realistic')
    setAvatarPreview(null)
    setAvatarPath(null)
    setVoiceMode('record')
    setRecordedAudio(null)
    setTtsText('')
    setUploadedAudio(null)
    setVoiceEffect('normal')
    setProcessedAudio(null)
    setAudioDuration(0)
    setGeneratedVideo(null)
    setVideoJobId(null)
    setVideoStatus(null)
  }, [])

  // Check if ready to generate
  const canGenerate = avatarPath && (processedAudio || recordedAudio || uploadedAudio || ttsText)

  return (
    <div className="min-h-screen">
      <Header onReset={resetAll} onOpenSettings={() => setSettingsOpen(true)} />

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 mb-6 text-red-200">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
              <button onClick={() => setError(null)} className="ml-auto hover:text-red-100">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Progress Steps */}
        <div className="flex items-center justify-center mb-8">
          {[
            { num: 1, label: 'Create Avatar' },
            { num: 2, label: 'Add Voice' },
            { num: 3, label: 'Generate Video' }
          ].map((step, index) => (
            <React.Fragment key={step.num}>
              <div
                className={`flex flex-col items-center cursor-pointer transition-all ${
                  currentStep >= step.num ? 'text-primary-400' : 'text-white/40'
                }`}
                onClick={() => setCurrentStep(step.num)}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg
                    ${currentStep >= step.num
                      ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white'
                      : 'bg-white/10 text-white/40'
                    }`}
                >
                  {currentStep > step.num ? (
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    step.num
                  )}
                </div>
                <span className="mt-2 text-sm font-medium">{step.label}</span>
              </div>
              {index < 2 && (
                <div
                  className={`w-24 h-1 mx-2 rounded ${
                    currentStep > step.num ? 'bg-gradient-to-r from-primary-500 to-accent-500' : 'bg-white/10'
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Avatar Creator */}
          <div className={`transition-all duration-300 ${currentStep === 1 ? 'ring-2 ring-primary-500/50' : ''}`}>
            <AvatarCreator
              description={avatarDescription}
              setDescription={setAvatarDescription}
              style={avatarStyle}
              setStyle={setAvatarStyle}
              preview={avatarPreview}
              setPreview={setAvatarPreview}
              avatarPath={avatarPath}
              setAvatarPath={setAvatarPath}
              onComplete={() => setCurrentStep(2)}
              setError={setError}
            />
          </div>

          {/* Right Column - Voice Section */}
          <div className={`space-y-6 transition-all duration-300 ${currentStep === 2 ? 'ring-2 ring-primary-500/50' : ''}`}>
            <VoiceRecorder
              mode={voiceMode}
              setMode={setVoiceMode}
              recordedAudio={recordedAudio}
              setRecordedAudio={setRecordedAudio}
              ttsText={ttsText}
              setTtsText={setTtsText}
              uploadedAudio={uploadedAudio}
              setUploadedAudio={setUploadedAudio}
              audioDuration={audioDuration}
              setAudioDuration={setAudioDuration}
              setError={setError}
            />

            <VoiceEffects
              selectedEffect={voiceEffect}
              setSelectedEffect={setVoiceEffect}
              audioSource={recordedAudio || uploadedAudio}
              processedAudio={processedAudio}
              setProcessedAudio={setProcessedAudio}
              setError={setError}
            />
          </div>
        </div>

        {/* Generate Button */}
        <div className="mt-8">
          <GenerateButton
            canGenerate={canGenerate}
            isGenerating={isGenerating}
            setIsGenerating={setIsGenerating}
            avatarDescription={avatarDescription}
            avatarStyle={avatarStyle}
            avatarPath={avatarPath}
            voiceMode={voiceMode}
            recordedAudio={recordedAudio}
            uploadedAudio={uploadedAudio}
            ttsText={ttsText}
            voiceEffect={voiceEffect}
            processedAudio={processedAudio}
            setGeneratedVideo={setGeneratedVideo}
            setVideoJobId={setVideoJobId}
            setVideoStatus={setVideoStatus}
            setCurrentStep={setCurrentStep}
            setError={setError}
          />
        </div>

        {/* Video Preview */}
        {(generatedVideo || videoJobId) && (
          <div className="mt-8">
            <VideoPreview
              video={generatedVideo}
              jobId={videoJobId}
              status={videoStatus}
              setVideo={setGeneratedVideo}
              setStatus={setVideoStatus}
              setError={setError}
            />
          </div>
        )}

        {/* Video Gallery */}
        <div className="mt-12">
          <VideoGallery />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-16 py-6">
        <div className="container mx-auto px-4 text-center text-white/40 text-sm">
          <p>Avatar Video Creator - Create 3-minute talking avatar videos</p>
          <p className="mt-1">Powered by AI Technology</p>
        </div>
      </footer>

      {/* Yorkie Helper - Bottom right corner */}
      <YorkieHelper
        currentStep={currentStep}
        isGenerating={isGenerating}
        videoReady={videoStatus === 'completed'}
      />

      {/* Settings Modal */}
      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}

export default App
