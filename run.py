"""
Quick start script for Derek Dashboard
The Christman AI Project
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")

    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ required")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check if .env file exists
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        issues.append("⚠️  .env file not found. Create one with your API keys")
    else:
        print("✓ .env file found")

    # Check essential Python modules
    required_modules = [
        ("requests", "HTTP requests"),
        ("dotenv", "Environment variables"),
    ]

    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {description} module available")
        except ImportError:
            issues.append(
                f"❌ {module_name} not installed. Run: pip install {module_name}"
            )

    # Check directories
    required_dirs = ["data", "logs"]
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_name}/ directory")
        else:
            print(f"✓ {dir_name}/ directory exists")

    if issues:
        print("\n⚠️  Setup Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False

    print("\n✓ Environment ready!")
    return True


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 Derek Dashboard - Quick Start")
    print("The Christman AI Project")
    print("=" * 60 + "\n")

    # Check environment
    if not check_environment():
        print("\n⚠️  Please fix environment issues before starting")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting Derek Dashboard...")
    print("=" * 60 + "\n")

    # Import and run main
    try:
        from main import main as dashboard_main

        dashboard_main()
    except KeyboardInterrupt:
        print("\n\n👋 Derek Dashboard stopped by user")
    except Exception as e:  # pragma: no cover - debug aid
        print(f"\n❌ Error starting dashboard: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
