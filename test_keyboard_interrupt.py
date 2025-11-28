"""
Simple test to verify keyboard interrupt detection works
Run this and press SPACE or ESC to test
"""

import sys
import time

# Test msvcrt import
try:
    import msvcrt
    HAS_MSVCRT = True
    print("✅ msvcrt imported successfully (Windows keyboard detection available)")
except ImportError:
    HAS_MSVCRT = False
    print("❌ msvcrt NOT available (not on Windows?)")
    sys.exit(1)

print("\n" + "="*60)
print("KEYBOARD INTERRUPT TEST")
print("="*60)
print("Press SPACE or ESC to test interrupt detection")
print("Press 'q' to quit")
print("Any other key will be shown but won't interrupt")
print("="*60 + "\n")

interrupt_detected = False
key_count = 0

while not interrupt_detected:
    # Check if key was pressed
    if msvcrt.kbhit():
        key = msvcrt.getch()
        key_count += 1

        print(f"[{key_count}] Key pressed: {key} (repr: {repr(key)})")

        # Check for quit
        if key == b'q':
            print("Quit requested")
            break

        # Check for SPACE (32) or ESC (27)
        if key in [b' ', b'\x1b']:
            print(f"🎉 INTERRUPT DETECTED! Key was: {repr(key)}")
            interrupt_detected = True
        else:
            print(f"   -> Not an interrupt key, continuing...")

    # Small delay to prevent CPU spinning
    time.sleep(0.1)

print("\n" + "="*60)
if interrupt_detected:
    print("✅ SUCCESS: Keyboard interrupt is working!")
else:
    print("Test ended (quit pressed)")
print("="*60)
