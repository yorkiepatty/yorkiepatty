import React from 'react'

function Header({ onReset }) {
  return (
    <header className="border-b border-white/10 backdrop-blur-lg bg-black/20 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Avatar Video Creator</h1>
              <p className="text-xs text-white/50">Create 3-minute talking avatar videos</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-white/50 text-sm hidden sm:block">
              Max duration: 3 minutes
            </span>
            <button
              onClick={onReset}
              className="btn-secondary text-sm py-2 px-4"
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Start Over
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
