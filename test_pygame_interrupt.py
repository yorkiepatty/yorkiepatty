"""
Test pygame audio playback with keyboard interrupt
This mimics what sunny_ultimate_voice.py does
"""

import sys
import time
import os

# Test msvcrt import
try:
    import msvcrt
    HAS_MSVCRT = True
    print("✅ msvcrt imported successfully")
except ImportError:
    HAS_MSVCRT = False
    print("❌ msvcrt NOT available")
    sys.exit(1)

# Test pygame import
try:
    import pygame
    print("✅ pygame imported successfully")
except ImportError:
    print("❌ pygame NOT installed. Install with: pip install pygame")
    sys.exit(1)

def check_keyboard_interrupt():
    """Check if user pressed a key to interrupt (SPACE or ESC)"""
    if msvcrt.kbhit():
        key = msvcrt.getch()
        print(f"\n[DEBUG] Key pressed: {key} (repr: {repr(key)})")
        # Check for SPACE (32) or ESC (27)
        if key in [b' ', b'\x1b']:
            print(f"[DEBUG] ✅ Interrupt key detected!")
            return True
        else:
            print(f"[DEBUG] ❌ Not an interrupt key (need SPACE or ESC)")
    return False

print("\n" + "="*60)
print("PYGAME AUDIO INTERRUPT TEST")
print("="*60)
print("This will play a test audio tone")
print("Press SPACE or ESC while it plays to interrupt")
print("="*60 + "\n")

# Generate a simple test tone using pygame
print("Initializing pygame mixer...")
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Create a simple beep sound
print("Generating test audio...")
from pygame import sndarray
import numpy as np

sample_rate = 22050
duration = 5  # 5 seconds
frequency = 440  # A4 note

# Generate sine wave
t = np.linspace(0, duration, int(sample_rate * duration))
wave = np.sin(2 * np.pi * frequency * t) * 32767
wave = wave.astype(np.int16)

# Create stereo array
stereo_wave = np.column_stack((wave, wave))

# Create sound from array
sound = sndarray.make_sound(stereo_wave)

print(f"Playing {duration}-second test tone at {frequency}Hz...")
print("💡 Press SPACE or ESC to interrupt!\n")

# Play the sound
channel = sound.play()

was_interrupted = False
loop_count = 0

# Wait for playback, checking for interrupts
while channel.get_busy():
    loop_count += 1

    if loop_count % 10 == 0:  # Print every second (10 loops at 0.1s each)
        print(f"[Loop {loop_count}] Still playing... (press SPACE/ESC to stop)")

    # Check keyboard interrupt
    if check_keyboard_interrupt():
        print("\n🛑 INTERRUPT! Stopping playback...")
        was_interrupted = True
        channel.stop()
        break

    # Small delay
    time.sleep(0.1)

pygame.mixer.quit()

print("\n" + "="*60)
print(f"Playback ended after {loop_count} loops (~{loop_count * 0.1:.1f} seconds)")
print(f"Was interrupted: {was_interrupted}")
if was_interrupted:
    print("✅ SUCCESS: Keyboard interrupt is working with pygame!")
else:
    print("⚠️  Audio played to completion (no interrupt detected)")
print("="*60)
