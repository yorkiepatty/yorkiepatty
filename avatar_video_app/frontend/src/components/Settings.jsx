import React, { useState, useEffect } from 'react'

function Settings({ isOpen, onClose }) {
  const [apiKey, setApiKey] = useState('')
  const [status, setStatus] = useState(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  // Check current status on mount
  useEffect(() => {
    if (isOpen) {
      checkStatus()
    }
  }, [isOpen])

  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/settings/status')
      const data = await response.json()
      setStatus(data)
    } catch (err) {
      console.error('Failed to check status:', err)
    }
  }

  const saveApiKey = async () => {
    if (!apiKey.trim()) {
      setMessage('Please enter an API key')
      return
    }

    if (!apiKey.startsWith('sk-')) {
      setMessage('API key should start with "sk-"')
      return
    }

    setSaving(true)
    setMessage('')

    try {
      const response = await fetch('http://localhost:8080/api/settings/apikey', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey })
      })

      const data = await response.json()

      if (data.success) {
        setMessage(data.message)
        setStatus({ openai_configured: true, key_preview: data.key_preview })
        setApiKey('')
      } else {
        setMessage(data.message || 'Failed to save API key')
      }
    } catch (err) {
      setMessage('Error connecting to server')
    } finally {
      setSaving(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-dark-800 rounded-2xl p-6 max-w-md w-full mx-4 border border-white/10"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-white">Settings</h2>
          <button onClick={onClose} className="text-white/60 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Current Status */}
        <div className="mb-6 p-4 rounded-xl bg-dark-700">
          <h3 className="text-sm font-semibold text-white/60 mb-2">OpenAI API Status</h3>
          {status?.openai_configured ? (
            <div className="flex items-center gap-2 text-green-400">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>Connected</span>
              <span className="text-white/40 text-sm ml-2">{status.key_preview}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-yellow-400">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>Not configured - using placeholder avatars</span>
            </div>
          )}
        </div>

        {/* API Key Input */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              OpenAI API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full px-4 py-3 rounded-xl bg-dark-700 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <p className="text-xs text-white/40 mt-2">
              Get your key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-primary-400 hover:underline">platform.openai.com/api-keys</a>
            </p>
          </div>

          {message && (
            <div className={`p-3 rounded-lg text-sm ${message.includes('saved') || message.includes('now generate') ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
              {message}
            </div>
          )}

          <button
            onClick={saveApiKey}
            disabled={saving}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-primary-500 to-accent-500 text-white font-semibold hover:opacity-90 disabled:opacity-50 transition-all"
          >
            {saving ? 'Saving...' : 'Save API Key'}
          </button>
        </div>

        <div className="mt-6 pt-4 border-t border-white/10 text-center">
          <p className="text-xs text-white/40">
            Your API key is stored in memory only and will need to be re-entered when you restart the server.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Settings
