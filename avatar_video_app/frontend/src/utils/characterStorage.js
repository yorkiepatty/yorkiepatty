/**
 * Character Storage Utility
 * Manages saved characters in localStorage
 */

const STORAGE_KEY = 'avatar_characters'

export const getCharacters = () => {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch (error) {
    console.error('Error loading characters:', error)
    return []
  }
}

export const saveCharacter = (character) => {
  try {
    const characters = getCharacters()
    const existingIndex = characters.findIndex(c => c.id === character.id)

    if (existingIndex >= 0) {
      // Update existing character
      characters[existingIndex] = character
    } else {
      // Add new character
      characters.push(character)
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(characters))
    return true
  } catch (error) {
    console.error('Error saving character:', error)
    return false
  }
}

export const deleteCharacter = (characterId) => {
  try {
    const characters = getCharacters()
    const filtered = characters.filter(c => c.id !== characterId)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered))
    return true
  } catch (error) {
    console.error('Error deleting character:', error)
    return false
  }
}

export const createCharacter = (name, avatarImage, voice = 'default', personality = 'neutral') => {
  return {
    id: Date.now().toString(36) + Math.random().toString(36).substr(2),
    name,
    avatarImage, // Base64 or URL
    voice,
    personality, // funny, serious, sarcastic, cheerful, grumpy
    createdAt: new Date().toISOString()
  }
}
