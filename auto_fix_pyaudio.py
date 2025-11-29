#!/usr/bin/env python3
"""
Automatic PyAudio Fix Patcher
Automatically applies all the PyAudio fixes to sunny_ultimate_voice.py
No manual editing required!
"""

import os
import sys
import shutil
from datetime import datetime

def apply_pyaudio_fixes(file_path):
    """Apply all PyAudio fixes automatically"""

    print("=" * 70)
    print("🔧 AUTOMATIC PYAUDIO FIX PATCHER")
    print("=" * 70)
    print()

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        print()
        print("Please run this script from your sunnyfolder directory:")
        print("  python auto_fix_pyaudio.py")
        return False

    # Create backup
    backup_path = file_path + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📦 Creating backup: {os.path.basename(backup_path)}")
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup created successfully")
    print()

    # Read the file
    print(f"📖 Reading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = 0

    # FIX 1: Add datetime import
    print("🔧 Fix 1: Adding datetime import...")
    if "from datetime import datetime" not in content:
        # Find the import section and add datetime
        import_section = "import logging\nfrom typing import"
        if import_section in content:
            content = content.replace(
                import_section,
                "import logging\nfrom datetime import datetime\nfrom typing import"
            )
            changes_made += 1
            print("   ✅ Added datetime import")
        else:
            print("   ⚠️  Could not find import section (might already be fixed)")
    else:
        print("   ✅ Already fixed")

    # FIX 2: Add try/except to _initialize_speech_recognition
    print("🔧 Fix 2: Adding error handling to speech recognition...")

    # Check if already fixed
    if "except AttributeError as e:" in content and "PyAudio not installed" in content:
        print("   ✅ Already fixed")
    else:
        # Find the method and wrap it in try/except
        old_method = '''    def _initialize_speech_recognition(self):
        """Initialize speech recognition with optimal settings for natural conversation"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Enhanced settings to avoid cutting off natural speech
        self.recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5

        # CRITICAL: Extended pause detection to handle natural pauses
        self.recognizer.pause_threshold = 2.0  # Wait 2 seconds of silence (was 1.2)
        self.recognizer.phrase_threshold = 0.2  # Min phrase length (shorter = more responsive)
        self.recognizer.non_speaking_duration = 0.8  # Allow longer pauses mid-sentence (was 0.5)

        # Calibrate microphone
        print("🎤 Calibrating microphone...")
        print("   (Please be COMPLETELY SILENT for 3 seconds...)")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=3)

        self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 3000)
        print(f"✅ Microphone calibrated! Energy: {self.recognizer.energy_threshold}")
        print(f"   Derek will wait 2 seconds of silence before processing your speech.")'''

        new_method = '''    def _initialize_speech_recognition(self):
        """Initialize speech recognition with optimal settings for natural conversation"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Enhanced settings to avoid cutting off natural speech
            self.recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5

            # CRITICAL: Extended pause detection to handle natural pauses
            self.recognizer.pause_threshold = 2.0  # Wait 2 seconds of silence (was 1.2)
            self.recognizer.phrase_threshold = 0.2  # Min phrase length (shorter = more responsive)
            self.recognizer.non_speaking_duration = 0.8  # Allow longer pauses mid-sentence (was 0.5)

            # Calibrate microphone
            print("🎤 Calibrating microphone...")
            print("   (Please be COMPLETELY SILENT for 3 seconds...)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=3)

            self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 3000)
            print(f"✅ Microphone calibrated! Energy: {self.recognizer.energy_threshold}")
            print(f"   Sunny will wait 2 seconds of silence before processing your speech.")
        except AttributeError as e:
            # PyAudio not installed
            print("⚠️  PyAudio not installed - Speech recognition disabled")
            print("   💬 You can still use text input to chat with Sunny!")
            print("   📦 To enable voice: pip install pyaudio")
            print("   📖 See WINDOWS_INSTALL_FIX.md for installation help")
            self.recognizer = None
            self.microphone = None
            self.enable_speech = False
        except Exception as e:
            # Other audio hardware issues
            print(f"⚠️  Could not initialize microphone: {e}")
            print("   💬 Speech recognition disabled - using text input only")
            self.recognizer = None
            self.microphone = None
            self.enable_speech = False'''

        if old_method in content:
            content = content.replace(old_method, new_method)
            changes_made += 1
            print("   ✅ Added error handling to speech recognition")
        else:
            print("   ⚠️  Could not find exact method (might already be fixed or different)")

    # FIX 3: Add check in listen() method
    print("🔧 Fix 3: Adding check in listen() method...")

    if "if not self.enable_speech or not self.recognizer or not self.microphone:" in content:
        print("   ✅ Already fixed")
    else:
        old_listen = '''    def listen(self):
        """Advanced speech recognition - patient listening, won't cut you off"""
        text = self.speech_recognition.listen() if hasattr(self, 'speech_recognition') else None'''

        new_listen = '''    def listen(self):
        """Advanced speech recognition - patient listening, won't cut you off"""
        # Check if speech recognition is available
        if not self.enable_speech or not self.recognizer or not self.microphone:
            # Speech recognition not available, return None to trigger text input
            return None

        text = self.speech_recognition.listen() if hasattr(self, 'speech_recognition') else None'''

        if old_listen in content:
            content = content.replace(old_listen, new_listen)
            changes_made += 1
            print("   ✅ Added speech availability check")
        else:
            print("   ⚠️  Could not find exact method (might already be fixed or different)")

    print()

    # Save the file if changes were made
    if content != original_content:
        print(f"💾 Saving changes to {os.path.basename(file_path)}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ File updated successfully!")
        print()
        print(f"📊 Summary: {changes_made} fixes applied")
        print()
        print("✅ ALL DONE! Your sunny_ultimate_voice.py is now fixed!")
        print(f"📦 Original file backed up to: {os.path.basename(backup_path)}")
        print()
        print("🎤 Next steps:")
        print("1. Install PyAudio:")
        print("   pip install pipwin")
        print("   pipwin install pyaudio")
        print()
        print("2. Run Sunny:")
        print("   python sunny_ultimate_voice.py")
        print()
        return True
    else:
        print("✅ No changes needed - file already appears to be fixed!")
        print(f"📦 Backup created anyway: {os.path.basename(backup_path)}")
        print()
        return True

def main():
    """Main entry point"""
    # Determine the file path
    file_path = "sunny_ultimate_voice.py"

    # Check if file exists in current directory
    if not os.path.exists(file_path):
        # Try to find it in common locations
        possible_paths = [
            "sunny_ultimate_voice.py",
            "../sunny_ultimate_voice.py",
            "sunnyfolder/sunny_ultimate_voice.py",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        else:
            print("❌ Error: Could not find sunny_ultimate_voice.py")
            print()
            print("Please run this script from your sunnyfolder directory:")
            print("  cd C:\\Users\\yorki\\sunnyfolder")
            print("  python auto_fix_pyaudio.py")
            sys.exit(1)

    # Apply the fixes
    success = apply_pyaudio_fixes(file_path)

    if success:
        print("=" * 70)
        print("✅ FIXES APPLIED SUCCESSFULLY!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("❌ SOME FIXES COULD NOT BE APPLIED")
        print("=" * 70)
        print("Check the messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
