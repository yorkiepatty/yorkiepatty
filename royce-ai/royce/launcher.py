"""
Royce Launcher
--------------
Main entry point for Royce AI Assistant on Windows 11.

Supports three modes:
1. Voice mode  — hands-free conversation via microphone
2. Text mode   — type in a terminal
3. Hybrid mode — voice + text combined (default)

Also handles:
- Windows 11 system tray integration
- Startup on boot (optional)
- Background news/horoscope scanning
"""

import sys
import os
import signal
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from royce.config import RoyceConfig
from royce.conversation import RoyceConversation
from royce.voice import RoyceVoice

# ─── Logging Setup ─────────────────────────────────────────────

log_file = RoyceConfig.LOGS_DIR / "royce.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("royce")


class RoyceLauncher:
    """Main launcher for Royce AI Assistant."""

    def __init__(self, mode: str = "text"):
        """
        Args:
            mode: "voice", "text", or "hybrid"
        """
        self.mode = mode
        self.running = False

        logger.info("=" * 60)
        logger.info("  ROYCE AI ASSISTANT")
        logger.info("  Confident. Smart. Caring.")
        logger.info("=" * 60)

        # Initialize core systems
        logger.info("Starting up...")
        self.conversation = RoyceConversation()
        self.voice = RoyceVoice()

        # Background tasks
        self._bg_threads = []

        logger.info("All systems initialized")

    def start(self):
        """Start Royce."""
        self.running = True

        # Register shutdown handler
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Start background tasks
        self._start_background_tasks()

        # Display greeting
        greeting = self.conversation.get_greeting()
        print(f"\nRoyce: {greeting}\n")
        if self.mode in ("voice", "hybrid"):
            self.voice.speak(greeting)

        # Enter main loop
        if self.mode == "voice":
            self._voice_loop()
        elif self.mode == "hybrid":
            self._hybrid_loop()
        else:
            self._text_loop()

    # ─── Main Loops ────────────────────────────────────────────────

    def _text_loop(self):
        """Text-only conversation loop."""
        print("(Type your messages. Say 'quit' or 'exit' to stop.)\n")

        while self.running:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "bye", "goodbye", "see ya"):
                    farewell = "Later. I'll be here when you need me."
                    print(f"\nRoyce: {farewell}\n")
                    break

                # Process and respond
                response = self.conversation.process(user_input)
                print(f"\nRoyce: {response}\n")

            except EOFError:
                break
            except KeyboardInterrupt:
                break

        self.shutdown()

    def _voice_loop(self):
        """Voice-only conversation loop."""
        print("(Listening... speak naturally. Say 'goodbye' to stop.)\n")

        while self.running:
            try:
                user_input = self.voice.listen()

                if not user_input:
                    continue

                print(f"You: {user_input}")

                if user_input.lower() in ("quit", "exit", "bye", "goodbye", "see ya"):
                    farewell = "Later. I'll be here when you need me."
                    print(f"Royce: {farewell}")
                    self.voice.speak(farewell)
                    break

                response = self.conversation.process(user_input)
                print(f"Royce: {response}\n")
                self.voice.speak(response)

            except KeyboardInterrupt:
                break

        self.shutdown()

    def _hybrid_loop(self):
        """Combined voice + text mode."""
        print("(Type or speak. Press Enter with no text to use voice. 'quit' to stop.)\n")

        while self.running:
            try:
                user_input = input("You (type or press Enter to speak): ").strip()

                if not user_input:
                    # Switch to voice for this input
                    print("(Listening...)")
                    user_input = self.voice.listen()
                    if not user_input:
                        print("(Didn't catch that. Try again or type it.)")
                        continue
                    print(f"You said: {user_input}")

                if user_input.lower() in ("quit", "exit", "bye", "goodbye", "see ya"):
                    farewell = "Later. I'll be here when you need me."
                    print(f"\nRoyce: {farewell}\n")
                    self.voice.speak(farewell)
                    break

                response = self.conversation.process(user_input)
                print(f"\nRoyce: {response}\n")
                self.voice.speak(response)

            except EOFError:
                break
            except KeyboardInterrupt:
                break

        self.shutdown()

    # ─── Background Tasks ──────────────────────────────────────────

    def _start_background_tasks(self):
        """Start background scanning for news and other periodic tasks."""

        def _periodic_news_scan():
            """Periodically check for breaking news."""
            while self.running:
                try:
                    time.sleep(RoyceConfig.NEWS_REFRESH_INTERVAL * 60)
                    if self.running:
                        self.conversation.news.get_breaking_news(count=5)
                        logger.info("Background news scan completed")
                except Exception as e:
                    logger.error(f"Background news scan error: {e}")

        news_thread = threading.Thread(target=_periodic_news_scan, daemon=True)
        news_thread.start()
        self._bg_threads.append(news_thread)

    # ─── Shutdown ──────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nRoyce: Alright, shutting down. Catch you later.\n")
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Clean shutdown of all systems."""
        self.running = False
        self.voice.stop_listening()
        self.conversation.shutdown()
        logger.info("Royce shut down cleanly")


# ─── Windows 11 Integration ───────────────────────────────────────

def create_startup_shortcut():
    """Create a Windows startup shortcut so Royce starts on boot."""
    try:
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        script_path = os.path.abspath(__file__)
        python_path = sys.executable

        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "Royce", 0, winreg.REG_SZ, f'"{python_path}" "{script_path}"')
        winreg.CloseKey(key)
        print("Royce will now start automatically when Windows boots.")
    except ImportError:
        print("Startup shortcut creation is only available on Windows.")
    except Exception as e:
        print(f"Couldn't create startup shortcut: {e}")


def remove_startup_shortcut():
    """Remove the Windows startup shortcut."""
    try:
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, "Royce")
        winreg.CloseKey(key)
        print("Royce will no longer start automatically.")
    except ImportError:
        print("This is only available on Windows.")
    except FileNotFoundError:
        print("Royce wasn't set to start automatically.")
    except Exception as e:
        print(f"Couldn't remove startup shortcut: {e}")


def send_windows_notification(title: str, message: str):
    """Send a Windows 11 toast notification."""
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=5, threaded=True)
    except ImportError:
        # Try PowerShell fallback
        try:
            import subprocess
            ps_script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
            $textNodes = $template.GetElementsByTagName("text")
            $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) > $null
            $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) > $null
            $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Royce").Show($toast)
            '''
            subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
        except Exception:
            pass


# ─── CLI Entry Point ──────────────────────────────────────────────

def main():
    """Main entry point — parse args and launch Royce."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Royce AI Assistant for Windows 11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m royce.launcher                  # Text mode (default)
  python -m royce.launcher --mode voice     # Voice-only mode
  python -m royce.launcher --mode hybrid    # Voice + text mode
  python -m royce.launcher --startup        # Add to Windows startup
  python -m royce.launcher --no-startup     # Remove from Windows startup
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["text", "voice", "hybrid"],
        default="text",
        help="Interaction mode (default: text)"
    )
    parser.add_argument(
        "--startup",
        action="store_true",
        help="Add Royce to Windows startup"
    )
    parser.add_argument(
        "--no-startup",
        action="store_true",
        help="Remove Royce from Windows startup"
    )

    args = parser.parse_args()

    if args.startup:
        create_startup_shortcut()
        return

    if args.no_startup:
        remove_startup_shortcut()
        return

    # Launch Royce
    royce = RoyceLauncher(mode=args.mode)
    royce.start()


if __name__ == "__main__":
    main()
