import React, { useState, useEffect } from 'react'
import { getCharacters, saveCharacter, createCharacter } from '../utils/characterStorage'

function ConversationMode({ onBack, onGenerate }) {
  const [mode, setMode] = useState(null) // 'ai' or 'scripted'
  const [characters, setCharacters] = useState([])
  const [character1, setCharacter1] = useState(null)
  const [character2, setCharacter2] = useState(null)
  const [script, setScript] = useState([]) // Array of { speaker: 1 or 2, text: string }
  const [currentInput, setCurrentInput] = useState('')
  const [isCreatingCharacter, setIsCreatingCharacter] = useState(null) // 1 or 2

  useEffect(() => {
    loadCharacters()
  }, [])

  const loadCharacters = () => {
    const saved = getCharacters()
    setCharacters(saved)
  }

  const handleCreateCharacter = (slot) => {
    setIsCreatingCharacter(slot)
  }

  const handleSaveNewCharacter = (name, avatarImage, voice, personality) => {
    const newChar = createCharacter(name, avatarImage, voice, personality)
    saveCharacter(newChar)
    loadCharacters()

    if (isCreatingCharacter === 1) {
      setCharacter1(newChar)
    } else {
      setCharacter2(newChar)
    }
    setIsCreatingCharacter(null)
  }

  const addDialogue = (speaker, text) => {
    if (!text.trim()) return
    setScript([...script, { speaker, text: text.trim() }])
    setCurrentInput('')
  }

  const removeDialogue = (index) => {
    setScript(script.filter((_, i) => i !== index))
  }

  const generateAIResponse = async (userText) => {
    // TODO: Call backend API to generate AI response
    // For now, simple placeholder
    const responses = [
      "That's interesting! Tell me more.",
      "I totally agree with that.",
      "Hmm, I'm not so sure about that.",
      "You've got a point there!",
      "That's hilarious! 😂"
    ]
    return responses[Math.floor(Math.random() * responses.length)]
  }

  const handleAddDialogue = async () => {
    if (!currentInput.trim()) return

    // Add Character 1's dialogue
    addDialogue(1, currentInput)

    // If AI mode, generate Character 2's response
    if (mode === 'ai') {
      const aiResponse = await generateAIResponse(currentInput)
      setTimeout(() => {
        setScript(prev => [...prev, { speaker: 2, text: aiResponse }])
      }, 500)
    }

    setCurrentInput('')
  }

  const handleGenerate = () => {
    if (!character1 || !character2 || script.length === 0) {
      alert('Please select both characters and add at least one dialogue!')
      return
    }

    onGenerate({
      mode,
      character1,
      character2,
      script
    })
  }

  // Mode selection screen
  if (!mode) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 py-8 px-4">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={onBack}
            className="mb-6 text-white/80 hover:text-white flex items-center gap-2"
          >
            ← Back
          </button>

          <h1 className="text-4xl font-bold text-white mb-4">
            Conversation Mode
          </h1>
          <p className="text-white/80 mb-8">
            Create conversations between two avatars!
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            {/* AI Mode */}
            <button
              onClick={() => setMode('ai')}
              className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 text-left hover:bg-white/20 transition-all group"
            >
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-2xl font-bold text-white mb-2">AI Conversation</h3>
              <p className="text-white/70 mb-4">
                You speak as one character, AI responds as the other.
                Great for improvisation!
              </p>
              <div className="text-primary-300 group-hover:text-primary-200">
                Start AI Conversation →
              </div>
            </button>

            {/* Scripted Mode */}
            <button
              onClick={() => setMode('scripted')}
              className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 text-left hover:bg-white/20 transition-all group"
            >
              <div className="text-4xl mb-4">📝</div>
              <h3 className="text-2xl font-bold text-white mb-2">Scripted Conversation</h3>
              <p className="text-white/70 mb-4">
                Write dialogue for both characters yourself.
                Perfect for comedy skits and storytelling!
              </p>
              <div className="text-accent-300 group-hover:text-accent-200">
                Start Scripted Conversation →
              </div>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Character creation modal
  if (isCreatingCharacter) {
    return (
      <CharacterCreator
        onSave={handleSaveNewCharacter}
        onCancel={() => setIsCreatingCharacter(null)}
      />
    )
  }

  // Character selection & script building
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <button
          onClick={() => setMode(null)}
          className="mb-6 text-white/80 hover:text-white flex items-center gap-2"
        >
          ← Change Mode
        </button>

        <h1 className="text-3xl font-bold text-white mb-2">
          {mode === 'ai' ? '🤖 AI Conversation' : '📝 Scripted Conversation'}
        </h1>
        <p className="text-white/70 mb-8">
          {mode === 'ai'
            ? 'You control Character 1, AI responds as Character 2'
            : 'Write dialogue for both characters'
          }
        </p>

        {/* Character Selection */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <CharacterSelector
            label="Character 1"
            character={character1}
            characters={characters}
            onSelect={setCharacter1}
            onCreate={() => handleCreateCharacter(1)}
          />
          <CharacterSelector
            label="Character 2"
            character={character2}
            characters={characters}
            onSelect={setCharacter2}
            onCreate={() => handleCreateCharacter(2)}
          />
        </div>

        {/* Script Builder */}
        {character1 && character2 && (
          <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-6 mb-6">
            <h3 className="text-xl font-bold text-white mb-4">Conversation Script</h3>

            {/* Script Display */}
            <div className="space-y-3 mb-6 max-h-96 overflow-y-auto">
              {script.length === 0 && (
                <p className="text-white/50 text-center py-8">
                  No dialogue yet. Start writing below!
                </p>
              )}
              {script.map((line, index) => (
                <div
                  key={index}
                  className={`flex items-start gap-3 ${
                    line.speaker === 1 ? 'justify-start' : 'justify-end'
                  }`}
                >
                  <div
                    className={`max-w-md p-4 rounded-2xl ${
                      line.speaker === 1
                        ? 'bg-primary-500/20 border border-primary-400/30'
                        : 'bg-accent-500/20 border border-accent-400/30'
                    }`}
                  >
                    <div className="font-bold text-white mb-1">
                      {line.speaker === 1 ? character1.name : character2.name}
                    </div>
                    <p className="text-white/90">{line.text}</p>
                  </div>
                  <button
                    onClick={() => removeDialogue(index)}
                    className="text-white/50 hover:text-red-400 text-sm mt-1"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>

            {/* Input Area */}
            <div className="space-y-3">
              {mode === 'scripted' && (
                <div className="flex gap-2">
                  <button
                    onClick={() => addDialogue(1, currentInput)}
                    disabled={!currentInput.trim()}
                    className="px-4 py-2 bg-primary-500 hover:bg-primary-600 disabled:bg-gray-600 disabled:opacity-50 rounded-lg text-white font-medium"
                  >
                    {character1.name} says
                  </button>
                  <button
                    onClick={() => addDialogue(2, currentInput)}
                    disabled={!currentInput.trim()}
                    className="px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:bg-gray-600 disabled:opacity-50 rounded-lg text-white font-medium"
                  >
                    {character2.name} says
                  </button>
                </div>
              )}

              <div className="flex gap-3">
                <input
                  type="text"
                  value={currentInput}
                  onChange={(e) => setCurrentInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddDialogue()}
                  placeholder={
                    mode === 'ai'
                      ? `What does ${character1.name} say?`
                      : 'Type dialogue...'
                  }
                  className="flex-1 px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:border-primary-400"
                />
                <button
                  onClick={handleAddDialogue}
                  disabled={!currentInput.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 disabled:from-gray-600 disabled:to-gray-600 disabled:opacity-50 rounded-lg text-white font-medium"
                >
                  {mode === 'ai' ? 'Add & AI Responds' : 'Add'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Generate Button */}
        {script.length > 0 && (
          <button
            onClick={handleGenerate}
            className="w-full py-4 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 rounded-xl text-white font-bold text-lg shadow-lg"
          >
            Generate Conversation Video
          </button>
        )}
      </div>
    </div>
  )
}

// Character Selector Component
function CharacterSelector({ label, character, characters, onSelect, onCreate }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-6">
      <h3 className="text-lg font-bold text-white mb-4">{label}</h3>

      {character ? (
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-400 to-accent-400 flex items-center justify-center text-2xl overflow-hidden">
            {character.avatarImage ? (
              <img src={character.avatarImage} alt={character.name} className="w-full h-full object-cover" />
            ) : (
              '👤'
            )}
          </div>
          <div className="flex-1">
            <div className="font-bold text-white">{character.name}</div>
            <div className="text-white/60 text-sm capitalize">{character.personality}</div>
          </div>
          <button
            onClick={() => onSelect(null)}
            className="text-white/70 hover:text-white"
          >
            Change
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full py-3 bg-white/5 border border-white/30 hover:bg-white/10 rounded-lg text-white"
          >
            Select Character
          </button>
          <button
            onClick={onCreate}
            className="w-full py-3 bg-primary-500 hover:bg-primary-600 rounded-lg text-white font-medium"
          >
            + Create New
          </button>

          {isOpen && characters.length > 0 && (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {characters.map(char => (
                <button
                  key={char.id}
                  onClick={() => {
                    onSelect(char)
                    setIsOpen(false)
                  }}
                  className="w-full p-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-left flex items-center gap-3"
                >
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-accent-400 flex items-center justify-center overflow-hidden">
                    {char.avatarImage ? (
                      <img src={char.avatarImage} alt={char.name} className="w-full h-full object-cover" />
                    ) : (
                      '👤'
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="text-white font-medium">{char.name}</div>
                    <div className="text-white/50 text-sm capitalize">{char.personality}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Character Creator Component
function CharacterCreator({ onSave, onCancel }) {
  const [name, setName] = useState('')
  const [personality, setPersonality] = useState('neutral')
  const [avatarImage, setAvatarImage] = useState('')

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setAvatarImage(event.target.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSave = () => {
    if (!name.trim()) {
      alert('Please enter a character name')
      return
    }
    onSave(name, avatarImage, 'default', personality)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 py-8 px-4 flex items-center justify-center">
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 max-w-md w-full">
        <h2 className="text-2xl font-bold text-white mb-6">Create Character</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-white mb-2">Character Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Detective Jones"
              className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:border-primary-400"
            />
          </div>

          <div>
            <label className="block text-white mb-2">Personality</label>
            <select
              value={personality}
              onChange={(e) => setPersonality(e.target.value)}
              className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white focus:outline-none focus:border-primary-400"
            >
              <option value="neutral">Neutral</option>
              <option value="funny">Funny</option>
              <option value="serious">Serious</option>
              <option value="sarcastic">Sarcastic</option>
              <option value="cheerful">Cheerful</option>
              <option value="grumpy">Grumpy</option>
            </select>
          </div>

          <div>
            <label className="block text-white mb-2">Avatar Image</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-primary-500 file:text-white hover:file:bg-primary-600"
            />
            {avatarImage && (
              <div className="mt-4">
                <img src={avatarImage} alt="Preview" className="w-32 h-32 rounded-full object-cover mx-auto" />
              </div>
            )}
          </div>

          <div className="flex gap-3 pt-4">
            <button
              onClick={onCancel}
              className="flex-1 py-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex-1 py-3 bg-primary-500 hover:bg-primary-600 rounded-lg text-white font-medium"
            >
              Save Character
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConversationMode
