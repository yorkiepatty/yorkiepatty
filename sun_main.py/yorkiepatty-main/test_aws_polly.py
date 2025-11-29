#!/usr/bin/env python3
"""
AWS Polly Voice Test for Sonny
Quick test to verify AWS Polly is working correctly
"""

import boto3
import os
import tempfile
import uuid
import subprocess
import platform
from dotenv import load_dotenv

load_dotenv()

def playsound(audio_file):
    """Play audio file using system-appropriate method"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_file], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", audio_file], check=True)
        elif system == "Windows":
            import winsound
            winsound.PlaySound(audio_file, winsound.SND_FILENAME)
        else:
            print(f"⚠️  Audio playback not supported on {system}")
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e}")

def test_aws_polly():
    """Test AWS Polly voice synthesis"""
    print("🎙️  Testing AWS Polly Voice Synthesis")
    print("=" * 50)
    
    try:
        # Initialize AWS Polly client
        polly = boto3.client('polly')
        print("✅ AWS Polly client initialized")
        
        # Test text
        test_text = "Hello! I'm Sonny, your coding mentor. I'm ready to help Yorkie learn programming!"
        
        # Available neural voices for testing
        test_voices = ["Matthew", "Joanna", "Stephen", "Ruth", "Kevin"]
        
        print(f"\n📝 Test text: {test_text}")
        print(f"🔊 Available neural voices: {', '.join(test_voices)}")
        
        # Test with Matthew (Sonny's default voice)
        print(f"\n🎯 Testing with Matthew (Sonny's default voice)...")
        
        response = polly.synthesize_speech(
            Text=test_text,
            OutputFormat='mp3',
            VoiceId='Matthew',
            Engine='neural'
        )
        
        # Save audio to temporary file
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"sonny_voice_test_{uuid.uuid4()}.mp3")
        
        with open(audio_file, 'wb') as f:
            f.write(response['AudioStream'].read())
        
        print(f"✅ Audio file created: {audio_file}")
        print("🔊 Playing audio...")
        
        # Play the audio
        playsound(audio_file)
        
        # Clean up
        try:
            os.remove(audio_file)
            print("🧹 Temporary file cleaned up")
        except:
            print("⚠️  Could not remove temporary file")
        
        print("\n✅ AWS Polly test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ AWS Polly test failed: {e}")
        
        # Check for common issues
        if "NoCredentialsError" in str(e):
            print("\n💡 AWS Credentials Issue:")
            print("   Make sure you have AWS credentials configured:")
            print("   - AWS CLI configured: `aws configure`")
            print("   - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("   - IAM role with Polly permissions")
            
        elif "InvalidParameterValue" in str(e):
            print("\n💡 Voice Parameter Issue:")
            print("   The voice 'Matthew' might not be available in your region")
            print("   Try a different voice like 'Joanna' or 'Stephen'")
            
        elif "UnauthorizedOperation" in str(e):
            print("\n💡 Permission Issue:")
            print("   Your AWS account needs Polly permissions")
            print("   Required permission: 'polly:SynthesizeSpeech'")
            
        return False

def check_aws_credentials():
    """Check if AWS credentials are properly configured"""
    print("🔐 Checking AWS Credentials...")
    
    # Check environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if aws_access_key and aws_secret_key:
        print("✅ AWS credentials found in environment variables")
        print(f"🌍 Region: {aws_region}")
        return True
    else:
        print("⚠️  AWS credentials not found in environment variables")
        print("💡 Checking AWS CLI configuration...")
        
        try:
            import subprocess
            result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ AWS CLI is configured")
                return True
            else:
                print("❌ AWS CLI not configured properly")
                return False
        except Exception:
            print("❌ AWS CLI not available or not configured")
            return False

def main():
    """Main test function"""
    print("🎯 Sonny AWS Polly Voice Test")
    print("=" * 60)
    print("Testing AWS Polly integration for Sonny's voice system\n")
    
    # Check credentials first
    if check_aws_credentials():
        print()
        # Run voice test
        if test_aws_polly():
            print("\n🎉 All tests passed! Sonny's voice is ready to go!")
        else:
            print("\n❌ Voice test failed. Check the issues above.")
    else:
        print("\n❌ Cannot test voice without AWS credentials.")
        print("\n💡 To set up AWS credentials:")
        print("1. Install AWS CLI: pip install awscli")
        print("2. Configure: aws configure")
        print("3. Or set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   export AWS_DEFAULT_REGION=us-east-1")

if __name__ == "__main__":
    main()