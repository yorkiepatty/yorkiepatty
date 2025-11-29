"""
Sunny Name Fix Patch Script
Fixes all references from Derek to Sunny
"""

import os
import re

def patch_file(filename, find_replace_pairs):
    """Apply find/replace to a file"""
    if not os.path.exists(filename):
        print(f"⚠️  {filename} not found, skipping...")
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = False
    
    for find, replace in find_replace_pairs:
        if find in content:
            content = content.replace(find, replace)
            changes_made = True
            print(f"✅ Fixed: {find[:50]}... → {replace[:50]}...")
    
    if changes_made:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {filename} patched successfully!\n")
        return True
    else:
        print(f"ℹ️  No changes needed in {filename}\n")
        return False

print("=" * 60)
print("🔧 Sunny Name Fix Patch")
print("=" * 60)
print()

# Fix 1: derek_manifest.yaml
print("📝 Patching derek_manifest.yaml...")
patch_file('derek_manifest.yaml', [
    ('name: "Derek C"', 'name: "Sunny"'),
    ("name: 'Derek C'", "name: 'Sunny'"),
])

# Fix 2: sunny_ultimate_voice.py
print("📝 Patching sunny_ultimate_voice.py...")
patch_file('sunny_ultimate_voice.py', [
    ('print(f"🗣️  Derek: {text}\\n")', 'print(f"🗣️  Sunny: {text}\\n")'),
    ("print(f'🗣️  Derek: {text}\\n')", "print(f'🗣️  Sunny: {text}\\n')"),
])

# Fix 3: main.py (if exists)
print("📝 Patching main.py...")
patch_file('main.py', [
    ("derek_ultimate_voice_module = derek_loader.get_module('derek_ultimate_voice')",
     "sunny_ultimate_voice_module = derek_loader.get_module('sunny_ultimate_voice')"),
    ('if derek_ultimate_voice_module:', 'if sunny_ultimate_voice_module:'),
    ('DerekUltimateVoice = derek_ultimate_voice_module.DerekUltimateVoice',
     'SunnyUltimateVoice = sunny_ultimate_voice_module.SunnyUltimateVoice'),
    ('POLLY_VOICES = derek_ultimate_voice_module.POLLY_VOICES',
     'POLLY_VOICES = sunny_ultimate_voice_module.POLLY_VOICES'),
    ('playsound = derek_ultimate_voice_module.playsound',
     'playsound = sunny_ultimate_voice_module.playsound'),
    ('if DerekUltimateVoice:', 'if SunnyUltimateVoice:'),
    ('derek_ultimate_voice = DerekUltimateVoice(',
     'sunny_ultimate_voice = SunnyUltimateVoice('),
    ('voice_id="matthew"', 'voice_id="Sunny"'),
    ('logger.info("✅ DerekUltimateVoice initialized',
     'logger.info("✅ SunnyUltimateVoice initialized'),
    ('logger.warning("⚠️ DerekUltimateVoice module not loaded")',
     'logger.warning("⚠️ SunnyUltimateVoice module not loaded")'),
    ('self.derek_ultimate_voice = derek_ultimate_voice',
     'self.sunny_ultimate_voice = sunny_ultimate_voice'),
    ('if self.derek_ultimate_voice:', 'if self.sunny_ultimate_voice:'),
    ('self.derek_ultimate_voice.speak', 'self.sunny_ultimate_voice.speak'),
    ('logger.info(f"🗣️ Derek says: {greeting}")',
     'logger.info(f"🗣️ Sunny says: {greeting}")'),
])

# Fix 4: .env file
print("📝 Checking .env file...")
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as f:
        env_content = f.read()
    
    if 'ELEVENLABS_VOICE_ID' not in env_content:
        # Find the ELEVENLABS_API_KEY line and add VOICE_ID after it
        if 'ELEVENLABS_API_KEY' in env_content:
            env_content = env_content.replace(
                'ELEVENLABS_API_KEY=',
                'ELEVENLABS_API_KEY='
            )
            # Add after the API key line
            lines = env_content.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if line.startswith('ELEVENLABS_API_KEY='):
                    new_lines.append('ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here')
            
            with open('.env', 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            print("✅ Added ELEVENLABS_VOICE_ID to .env\n")
        else:
            print("⚠️  ELEVENLABS_API_KEY not found in .env\n")
    else:
        print("ℹ️  ELEVENLABS_VOICE_ID already exists in .env\n")
else:
    print("⚠️  .env file not found\n")

print("=" * 60)
print("✅ Patch complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Edit .env and add your real ELEVENLABS_API_KEY")
print("2. Edit .env and add your real ELEVENLABS_VOICE_ID")
print("3. Edit .env and add your real ANTHROPIC_API_KEY")
print("4. Restart Sunny: python sunny_ultimate_voice.py")
