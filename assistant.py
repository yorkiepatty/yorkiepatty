import json
import os
import speech_recognition as sr
import requests
import pygame
import io
from dotenv import load_dotenv
from pathlib import Path
import threading
import time
from pynput import keyboard
import webbrowser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3
from botocore.exceptions import ClientError
import base64
import mimetypes
try:
    from googlesearch import search
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("⚠️  Web search not available. Install with: pip install googlesearch-python")
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("⚠️  Video generation not available. Install with: pip install replicate")
import re

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # AWS Services (no API keys needed - uses AWS credentials)
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    USE_AWS_BEDROCK = True
    USE_AWS_POLLY = True
    
    # AWS Polly voice settings
    POLLY_VOICE_ID = os.getenv('POLLY_VOICE_ID', 'Joanna')
    POLLY_ENGINE = os.getenv('POLLY_ENGINE', 'neural')
    
    # AI Image/Video generation APIs
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY')
    
    # Email configuration (optional)
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    
    MEMORY_FILE = 'conversation_memory.json'
    PROJECTS_FILE = 'projects_memory.json'
    MAX_HISTORY_MESSAGES = 100
    MAX_TOKENS = 2000
    # Use the cross-region inference profile for Claude 3.5 Sonnet v2
    BEDROCK_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Speech recognition settings
    ENERGY_THRESHOLD = 2500
    LISTEN_TIMEOUT = 15
    PHRASE_TIME_LIMIT = 60
    
    # Interrupt detection settings
    INTERRUPT_ENERGY_THRESHOLD = 5000
    INTERRUPT_CHECK_INTERVAL = 0.3

print(f"Configuration loaded:")
print(f"  AWS Bedrock: Enabled (Region: {Config.AWS_REGION})")
print(f"  Bedrock Model: {Config.BEDROCK_MODEL}")
print(f"  AWS Polly: Enabled (Voice: {Config.POLLY_VOICE_ID}, Engine: {Config.POLLY_ENGINE})")
print(f"  Image Generation: {'Enabled (DALL-E)' if Config.OPENAI_API_KEY else 'Disabled'}")
print(f"  Video Generation: {'Enabled (Replicate)' if Config.REPLICATE_API_KEY and REPLICATE_AVAILABLE else 'Disabled'}")
print(f"  Vision: Enabled (Claude can see images and videos)")
print(f"  File Reading: Enabled (txt, pdf, docx, csv, xlsx, json, code files)")
print(f"  Video Processing: Enabled (mp4, avi, mov, mkv, webm)")
print(f"  Audio Processing: Enabled (mp3, wav, ogg, m4a)")
print(f"  Web Reading: Enabled (can read any webpage)")

# Initialize pygame mixer for audio
try:
    pygame.mixer.init()
    print("  Pygame mixer: Initialized")
except Exception as e:
    print(f"  Pygame init error: {e}")
    raise

# Initialize conversation memory
conversation_history = []
projects = {}


class FileHandler:
    """Handles reading and processing various file types and web pages"""
    
    # Supported file extensions
    TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h', '.css', 
                      '.html', '.xml', '.json', '.yaml', '.yml', '.sh', '.bat', '.log'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc'}
    DATA_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}
    
    @staticmethod
    def read_webpage(url):
        """Read and extract text content from a webpage"""
        try:
            from bs4 import BeautifulSoup
            
            if not url.startswith('http'):
                url = 'https://' + url
            
            print(f"\n🌐 Reading webpage: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Get title
            title = soup.title.string if soup.title else "No title"
            
            result = f"Webpage: {title}\nURL: {url}\n\n{text[:8000]}"  # Limit to 8000 chars
            
            print(f"✅ Successfully read webpage ({len(text)} characters)")
            return True, result
            
        except ImportError:
            return False, "Webpage reading requires BeautifulSoup4. Install with: pip install beautifulsoup4"
        except requests.exceptions.Timeout:
            return False, "Request timed out - webpage took too long to load"
        except requests.exceptions.RequestException as e:
            return False, f"Error fetching webpage: {str(e)}"
        except Exception as e:
            return False, f"Error reading webpage: {str(e)}"
    
    @staticmethod
    def can_read(file_path):
        """Check if file type is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in (FileHandler.TEXT_EXTENSIONS | 
                      FileHandler.IMAGE_EXTENSIONS | 
                      FileHandler.DOCUMENT_EXTENSIONS | 
                      FileHandler.DATA_EXTENSIONS |
                      FileHandler.VIDEO_EXTENSIONS |
                      FileHandler.AUDIO_EXTENSIONS)
    
    @staticmethod
    def read_text_file(file_path):
        """Read plain text or code files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return True, content
            except Exception as e:
                return False, f"Error reading file: {str(e)}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    @staticmethod
    def read_pdf(file_path):
        """Read PDF files"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text())
                return True, '\n\n'.join(text)
        except ImportError:
            return False, "PDF reading requires PyPDF2. Install with: pip install PyPDF2"
        except Exception as e:
            return False, f"Error reading PDF: {str(e)}"
    
    @staticmethod
    def read_docx(file_path):
        """Read Word documents"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return True, '\n'.join(text)
        except ImportError:
            return False, "DOCX reading requires python-docx. Install with: pip install python-docx"
        except Exception as e:
            return False, f"Error reading DOCX: {str(e)}"
    
    @staticmethod
    def read_csv(file_path):
        """Read CSV files"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            summary = f"CSV File: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            summary += f"Columns: {', '.join(df.columns)}\n\n"
            summary += f"First 10 rows:\n{df.head(10).to_string()}"
            return True, summary
        except ImportError:
            return False, "CSV reading requires pandas. Install with: pip install pandas"
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}"
    
    @staticmethod
    def read_excel(file_path):
        """Read Excel files"""
        try:
            import pandas as pd
            xl = pd.ExcelFile(file_path)
            text = [f"Excel file with sheets: {', '.join(xl.sheet_names)}\n"]
            
            for sheet in xl.sheet_names[:3]:  # Read first 3 sheets
                df = pd.read_excel(xl, sheet_name=sheet)
                text.append(f"\n--- Sheet: {sheet} ---")
                text.append(f"{df.shape[0]} rows, {df.shape[1]} columns")
                text.append(f"Columns: {', '.join(df.columns)}")
                text.append(f"\nFirst 5 rows:\n{df.head(5).to_string()}\n")
            
            return True, '\n'.join(text)
        except ImportError:
            return False, "Excel reading requires pandas and openpyxl. Install with: pip install pandas openpyxl"
        except Exception as e:
            return False, f"Error reading Excel: {str(e)}"
    
    @staticmethod
    def encode_image_base64(file_path):
        """Encode image to base64 for Claude vision"""
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Get media type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return True, base64_image, mime_type
        except Exception as e:
            return False, None, f"Error encoding image: {str(e)}"
    
    @staticmethod
    def process_video(file_path, num_frames=5):
        """Extract frames and audio transcript from video"""
        try:
            import cv2
            
            print(f"\n🎬 Processing video: {file_path}")
            
            video = cv2.VideoCapture(str(file_path))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Extract evenly spaced frames
            frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]
            frames = []
            
            for idx in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = video.read()
                if ret:
                    # Convert to RGB and encode
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    success, buffer = cv2.imencode('.jpg', frame_rgb)
                    if success:
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        frames.append(frame_base64)
            
            video.release()
            
            # Try to extract audio and transcribe
            audio_transcript = FileHandler.extract_audio_transcript(file_path)
            
            result = {
                'frames': frames,
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'audio_transcript': audio_transcript
            }
            
            print(f"✅ Extracted {len(frames)} frames from video ({duration:.1f}s)")
            if audio_transcript:
                print(f"✅ Extracted audio transcript")
            
            return True, result
            
        except ImportError:
            return False, "Video processing requires opencv-python. Install with: pip install opencv-python"
        except Exception as e:
            return False, f"Error processing video: {str(e)}"
    
    @staticmethod
    def extract_audio_transcript(video_path):
        """Extract audio from video and transcribe it"""
        try:
            import moviepy.editor as mp
            import speech_recognition as sr
            import tempfile
            
            # Extract audio
            video = mp.VideoFileClip(str(video_path))
            
            # Create temp audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                video.audio.write_audiofile(temp_audio_path, logger=None)
            
            video.close()
            
            # Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    transcript = recognizer.recognize_google(audio_data)
                    os.unlink(temp_audio_path)  # Clean up
                    return transcript
                except sr.UnknownValueError:
                    os.unlink(temp_audio_path)
                    return None
                except sr.RequestError:
                    os.unlink(temp_audio_path)
                    return None
                    
        except ImportError:
            return None  # moviepy not installed
        except Exception as e:
            print(f"Audio extraction warning: {e}")
            return None
    
    @staticmethod
    def process_audio(file_path):
        """Transcribe audio file"""
        try:
            import speech_recognition as sr
            
            print(f"\n🎵 Processing audio: {file_path}")
            print(f"   File exists: {Path(file_path).exists()}")
            print(f"   File size: {Path(file_path).stat().st_size / 1024:.2f} KB")
            
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300  # Lower threshold for recordings
            
            # Convert to WAV if needed
            converted = False
            original_path = file_path
            if not file_path.lower().endswith('.wav'):
                try:
                    from pydub import AudioSegment
                    print("   Converting to WAV format...")
                    
                    # Load audio file
                    audio = AudioSegment.from_file(file_path)
                    
                    # Normalize audio
                    audio = audio.set_channels(1)  # Mono
                    audio = audio.set_frame_rate(16000)  # Standard rate
                    
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        temp_wav_path = temp_wav.name
                        audio.export(temp_wav_path, format='wav')
                        file_path = temp_wav_path
                        converted = True
                        print(f"   ✅ Converted to: {temp_wav_path}")
                        
                except ImportError:
                    print("   ⚠️  pydub not installed, trying direct processing...")
                    return False, "Audio conversion requires pydub and ffmpeg. Install with: pip install pydub ffmpeg-python"
                except Exception as e:
                    print(f"   ❌ Conversion error: {e}")
                    return False, f"Audio conversion failed: {str(e)}"
            
            # Transcribe audio
            try:
                with sr.AudioFile(file_path) as source:
                    print("   Reading audio file...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                    print(f"   Audio duration: ~{len(audio_data.frame_data) / (audio_data.sample_rate * audio_data.sample_width):.1f} seconds")
                    
                    print("   Transcribing with Google Speech Recognition...")
                    transcript = recognizer.recognize_google(audio_data)
                    
                    # Clean up temp file
                    if converted:
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                    
                    print(f"   ✅ Transcribed audio successfully!")
                    print(f"   Transcript: {transcript[:100]}...")
                    return True, transcript
                    
            except sr.UnknownValueError:
                print("   ❌ Could not understand audio - speech may be unclear")
                if converted:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                return True, "[Audio file processed but speech was unclear or silent]"
                
            except sr.RequestError as e:
                print(f"   ❌ Transcription service error: {e}")
                if converted:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                return False, f"Transcription service error: {e}"
                    
        except Exception as e:
            print(f"   ❌ Error processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Error processing audio: {str(e)}"
    
    @staticmethod
    def read_file(file_path):
        """Universal file reader - detects type and reads accordingly"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, None, f"File not found: {file_path}"
        
        ext = file_path.suffix.lower()
        
        # Video files
        if ext in FileHandler.VIDEO_EXTENSIONS:
            success, data = FileHandler.process_video(file_path)
            if success:
                return True, 'video', data
            else:
                return False, None, data  # Error message
        
        # Audio files
        elif ext in FileHandler.AUDIO_EXTENSIONS:
            success, transcript = FileHandler.process_audio(file_path)
            if success:
                return True, 'audio', transcript
            else:
                return False, None, transcript  # Error message
        
        # Image files
        elif ext in FileHandler.IMAGE_EXTENSIONS:
            success, data, mime_type = FileHandler.encode_image_base64(file_path)
            if success:
                return True, 'image', {'data': data, 'mime_type': mime_type, 'filename': file_path.name}
            else:
                return False, None, mime_type  # Error message
        
        # Text/code files
        elif ext in FileHandler.TEXT_EXTENSIONS or ext == '.json':
            success, content = FileHandler.read_text_file(file_path)
            return success, 'text', content
        
        # PDF
        elif ext == '.pdf':
            success, content = FileHandler.read_pdf(file_path)
            return success, 'text', content
        
        # Word documents
        elif ext in {'.docx', '.doc'}:
            success, content = FileHandler.read_docx(file_path)
            return success, 'text', content
        
        # CSV
        elif ext == '.csv':
            success, content = FileHandler.read_csv(file_path)
            return success, 'text', content
        
        # Excel
        elif ext in {'.xlsx', '.xls'}:
            success, content = FileHandler.read_excel(file_path)
            return success, 'text', content
        
        else:
            return False, None, f"Unsupported file type: {ext}"


def extract_file_path(user_input):
    """Extract file path from user input or prompt for it"""
    # Method 1: Look for quoted paths
    quoted = re.findall(r'["\']([^"\']+)["\']', user_input)
    if quoted:
        for path in quoted:
            if Path(path).exists():
                return path
    
    # Method 2: Look for files with extensions
    file_patterns = re.findall(r'[\w\-./\\]+\.\w+', user_input)
    if file_patterns:
        for pattern in file_patterns:
            # Try as-is
            if Path(pattern).exists():
                return pattern
            # Try in current directory
            current_dir_path = Path.cwd() / pattern
            if current_dir_path.exists():
                return str(current_dir_path)
    
    # Method 3: Look for single word that might be a filename
    words = user_input.split()
    for word in words:
        # Clean up punctuation
        clean_word = word.strip('.,!?;:')
        if '.' in clean_word:
            if Path(clean_word).exists():
                return clean_word
            # Try in current directory
            current_dir_path = Path.cwd() / clean_word
            if current_dir_path.exists():
                return str(current_dir_path)
    
    # Method 4: Ask user for the path
    print("\n🔍 I couldn't detect a file path. Please enter the file path:")
    print("   (You can drag and drop the file here, or type the path)")
    file_path = input("File path: ").strip().strip('"\'')
    
    if file_path and Path(file_path).exists():
        return file_path
    elif file_path:
        # Try in current directory
        current_dir_path = Path.cwd() / file_path
        if current_dir_path.exists():
            return str(current_dir_path)
        print(f"❌ File not found: {file_path}")
        return None
    
    return None


class ActionHandler:
    """Handles external actions like web search, YouTube, email, image/video generation, etc."""
    
    @staticmethod
    def generate_image(prompt, service="openai"):
        """Generate AI image using DALL-E or Stable Diffusion"""
        output_dir = Path('generated_images')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            if service == "openai" and Config.OPENAI_API_KEY:
                print(f"\n🎨 Generating image with DALL-E: {prompt}")
                
                headers = {
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "quality": "standard"
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    image_url = result['data'][0]['url']
                    
                    img_response = requests.get(image_url)
                    filename = output_dir / f"dalle_{timestamp}.png"
                    
                    with open(filename, 'wb') as f:
                        f.write(img_response.content)
                    
                    print(f"✅ Image saved to: {filename}")
                    webbrowser.open(str(filename.absolute()))
                    
                    return True, str(filename)
                else:
                    error_msg = response.json().get('error', {}).get('message', response.text)
                    print(f"❌ DALL-E error: {response.status_code} - {error_msg}")
                    return False, f"Failed to generate image: {error_msg}"
            
            elif service == "replicate" and Config.REPLICATE_API_KEY and REPLICATE_AVAILABLE:
                print(f"\n🎨 Generating image with Stable Diffusion: {prompt}")
                
                os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_KEY
                
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={"prompt": prompt}
                )
                
                image_url = output[0] if isinstance(output, list) else output
                img_response = requests.get(image_url)
                filename = output_dir / f"sdxl_{timestamp}.png"
                
                with open(filename, 'wb') as f:
                    f.write(img_response.content)
                
                print(f"✅ Image saved to: {filename}")
                webbrowser.open(str(filename.absolute()))
                
                return True, str(filename)
            
            else:
                msg = "No image generation API configured. Set OPENAI_API_KEY or REPLICATE_API_KEY in .env"
                print(f"⚠️  {msg}")
                return False, msg
            
        except Exception as e:
            print(f"❌ Image generation error: {e}")
            return False, f"Error: {str(e)}"  
    try:
        if audio_path:
        print(f"\n🎬 Creating lip-sync animation with audio file")
        print(f"   Image: {image_path}")
        print(f"   Audio: {audio_path}")
    else:
        print(f"\n🎬 Creating talking animation with text")
        print(f"   Image: {image_path}")
        print(f"   Text: {audio_text}")
    
    print("⏳ This may take 2-3 minutes...")
    
    os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_KEY
    
    # Prepare input - use file handles instead of raw bytes
    model_input = {
        "preprocess": "full",
        "still_mode": False,
        "use_enhancer": True,
        "face_model_resolution": "256"
    }
    
    # Open image file as file handle
    with open(image_path, 'rb') as img_file:
        if audio_path:
            # Use recorded audio file - open as file handle
            with open(audio_path, 'rb') as audio_file:
                model_input["source_image"] = img_file
                model_input["driven_audio"] = audio_file
                
                # Use SadTalker for lip-sync
                print("   Running lip-sync model...")
                output = replicate.run(
                    "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a35248d584878dc4fe40c6e6c1a1dae8b3aee09cdfd1",
                    input=model_input
                )
        else:
            # Use text-to-speech
            model_input["source_image"] = img_file
            model_input["driven_audio"] = audio_text
            
            print("   Running lip-sync model with text-to-speech...")
            output = replicate.run(
                "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a35248d584878dc4fe40c6e6c1a1dae8b3aee09cdfd1",
                input=model_input
            )
    
    @staticmethod
    def animate_image_talking(image_path, audio_path=None, audio_text=None):
        """Animate image with lip-sync using recorded audio or text-to-speech"""
        if not Config.REPLICATE_API_KEY:
            return False, "Talking animation requires REPLICATE_API_KEY in .env"
        
        if not REPLICATE_AVAILABLE:
            return False, "Install replicate package: pip install replicate"
        
        # Check if image exists
        if not Path(image_path).exists():
            return False, f"Image not found: {image_path}"
        
        # Check if we have audio file or text
        if audio_path and not Path(audio_path).exists():
            return False, f"Audio file not found: {audio_path}"
        
        if not audio_path and not audio_text:
            audio_text = "Hello, this is a test."
        
        output_dir = Path('generated_videos')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
           # Handle output
            if isinstance(output, str):
                video_url = output
            elif isinstance(output, list):
                video_url = output[0] if output else None
            elif hasattr(output, 'url'):
                video_url = output.url
            else:
                try:
                    video_url = str(output)
                except:
                    return False, f"Unexpected output format: {type(output)}"
            
            if not video_url:
                return False, "No video URL returned from API"
            
            print(f"📥 Downloading lip-synced video from: {video_url}")
            
            # Download video
            video_response = requests.get(video_url, timeout=300)
            
            if video_response.status_code != 200:
                return False, f"Failed to download video: {video_response.status_code}"
            
            filename = output_dir / f"talking_{timestamp}.mp4"
            with open(filename, 'wb') as f:
                f.write(video_response.content)
            
            print(f"✅ Talking video saved to: {filename}")
            webbrowser.open(str(filename.absolute()))
            
            return True, str(filename)
            
        except Exception as e:
            import traceback
            print(f"❌ Talking animation error: {e}")
            print(traceback.format_exc())
            return False, f"Error: {str(e)}"
            
    @staticmethod
    def generate_video(prompt, duration=3):       
        """Generate AI video using Replicate"""
        if not Config.REPLICATE_API_KEY:
            return False, "Video generation requires REPLICATE_API_KEY in .env"
        
        if not REPLICATE_AVAILABLE:
            return False, "Install replicate package: pip install replicate"
        
        output_dir = Path('generated_videos')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
         
        try:
            print(f"\n🎬 Generating video: {prompt}")
            print("⏳ This may take 1-2 minutes...")
            
            os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_KEY
            
            output = replicate.run(
                "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",
                input={
                    "prompt": prompt,
                    "num_frames": duration * 8,
                    "guidance_scale": 7.5,
                }
            )
            
            if isinstance(output, list):
                video_url = output[0]
            elif isinstance(output, str):
                video_url = output
            elif hasattr(output, 'url'):
                video_url = output.url
            else:
                try:
                    video_url = str(output)
                except:
                    return False, f"Unexpected output format: {type(output)}"
            
            print(f"📥 Downloading video from: {video_url}")
            
            video_response = requests.get(video_url, timeout=300)
            
            if video_response.status_code != 200:
                return False, f"Failed to download video: {video_response.status_code}"
            
            filename = output_dir / f"video_{timestamp}.mp4"
            with open(filename, 'wb') as f:
                f.write(video_response.content)
            
            print(f"✅ Video saved to: {filename}")
            webbrowser.open(str(filename.absolute()))
            
            return True, str(filename)
            
        except Exception as e:
            import traceback
            print(f"❌ Video generation error: {e}")
            print(traceback.format_exc())
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def web_search(query, num_results=5):
        """Search the web and return results"""
        if not SEARCH_AVAILABLE:
            return ["Web search not available - install googlesearch-python"]
        
        try:
            print(f"\n🔍 Searching for: {query}")
            results = []
            for url in search(query, num_results=num_results, lang="en"):
                results.append(url)
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    @staticmethod
    def play_youtube(query):
        """Search and play YouTube video"""
        try:
            print(f"\n🎵 Searching YouTube for: {query}")
            search_query = query.replace(' ', '+')
            youtube_search_url = f"https://www.youtube.com/results?search_query={search_query}"
            
            response = requests.get(youtube_search_url)
            video_ids = re.findall(r"watch\?v=(\S{11})", response.text)
            
            if video_ids:
                video_url = f"https://www.youtube.com/watch?v={video_ids[0]}"
                print(f"🎥 Opening: {video_url}")
                webbrowser.open(video_url)
                return True, video_url
            else:
                print("No videos found")
                return False, None
        except Exception as e:
            print(f"YouTube error: {e}")
            return False, None
    
    @staticmethod
    def send_email(to_address, subject, body):
        """Send an email"""
        if not Config.EMAIL_ADDRESS or not Config.EMAIL_PASSWORD:
            return False, "Email not configured. Please set EMAIL_ADDRESS and EMAIL_PASSWORD in .env"
        
        try:
            print(f"\n📧 Sending email to: {to_address}")
            
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_ADDRESS
            msg['To'] = to_address
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            print("✅ Email sent successfully")
            return True, "Email sent!"
        except Exception as e:
            print(f"Email error: {e}")
            return False, f"Failed to send email: {str(e)}"
    
    @staticmethod
    def open_website(url):
        """Open a website in browser"""
        try:
            if not url.startswith('http'):
                url = 'https://' + url
            print(f"\n🌐 Opening: {url}")
            webbrowser.open(url)
            return True
        except Exception as e:
            print(f"Browser error: {e}")
            return False
    
    @staticmethod
    def get_weather(city):
        """Get weather information"""
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                current = data['current_condition'][0]
                weather_desc = current['weatherDesc'][0]['value']
                temp_f = current['temp_F']
                feels_like = current['FeelsLikeF']
                return f"In {city}, it's {weather_desc}, {temp_f}°F (feels like {feels_like}°F)"
            return None
        except Exception as e:
            print(f"Weather error: {e}")
            return None


class ConversationManager:
    """Manages conversation history and memory"""
    
    @staticmethod
    def save_memory():
        """Save conversation history to file"""
        try:
            with open(Config.MEMORY_FILE, 'w') as f:
                json.dump(conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    @staticmethod
    def save_projects():
        """Save projects to file"""
        global projects
        try:
            with open(Config.PROJECTS_FILE, 'w') as f:
                json.dump(projects, f, indent=2)
        except Exception as e:
            print(f"Error saving projects: {e}")

    @staticmethod
    def load_memory():
        """Load conversation history from file"""
        global conversation_history
        try:
            if Path(Config.MEMORY_FILE).exists():
                with open(Config.MEMORY_FILE, 'r') as f:
                    conversation_history = json.load(f)
                print(f"✅ Loaded {len(conversation_history)} previous messages")
            else:
                conversation_history = []
                print("Starting fresh conversation")
        except Exception as e:
            print(f"Error loading memory: {e}")
            conversation_history = []
    
    @staticmethod
    def load_projects():
        """Load projects from file"""
        global projects
        try:
            if Path(Config.PROJECTS_FILE).exists():
                with open(Config.PROJECTS_FILE, 'r') as f:
                    projects = json.load(f)
                print(f"✅ Loaded {len(projects)} projects")
            else:
                projects = {}
        except Exception as e:
            print(f"Error loading projects: {e}")
            projects = {}

    @staticmethod
    def clear_memory():
        """Clear conversation history"""
        global conversation_history
        conversation_history = []
        try:
            if Path(Config.MEMORY_FILE).exists():
                os.remove(Config.MEMORY_FILE)
            print("Memory cleared")
        except Exception as e:
            print(f"Error clearing memory: {e}")
    
    @staticmethod
    def trim_history():
        """Trim conversation history to prevent overflow"""
        global conversation_history
        if len(conversation_history) > Config.MAX_HISTORY_MESSAGES:
            conversation_history = conversation_history[-Config.MAX_HISTORY_MESSAGES:]
            ConversationManager.save_memory()
    
    @staticmethod
    def get_context_summary():
        """Get a summary of all projects"""
        global projects
        summary = []
        
        if projects:
            summary.append("ACTIVE PROJECTS:")
            for name, details in projects.items():
                summary.append(f"- {name}: {details.get('details', 'No description')}")
        
        return "\n".join(summary) if summary else "No active projects."


class BedrockClient:
    """Handles communication with AWS Bedrock (Claude) with vision support"""
    
    SYSTEM_PROMPT = """You are Sunny, a helpful but sassy AI voice assistant with expert coding abilities, action capabilities, and now VISION and FILE READING abilities!

IMPORTANT RULES:
- Never use asterisks for actions like *laughs* or *sighs*
- Keep responses conversational and concise since this is voice chat
- Be a bit cheeky and confident in your responses
- Use "I'm Sunny" when introducing yourself
- No emojis or special formatting - just natural speech
- Keep responses under 3-4 sentences unless asked for more detail
- If interrupted, acknowledge it briefly and ask what they need

MEMORY & CONTEXT:
- You have FULL memory of all our previous conversations - use it!
- Remember project details, preferences, and past discussions
- Reference previous conversations naturally when relevant
- Track ongoing projects and their progress

NEW CAPABILITIES - VISION, VIDEO, AUDIO, FILE READING & WEB READING:
- YOU CAN NOW SEE IMAGES! When shown a picture, describe what you see naturally
- YOU CAN WATCH VIDEOS! Analyze video frames and listen to audio transcripts
- YOU CAN LISTEN TO AUDIO! Transcribe and understand audio files
- YOU CAN READ FILES! Including: text files, PDFs, Word docs, Excel, CSV, code files, and more
- YOU CAN READ WEBPAGES! Extract and analyze content from any URL
- When analyzing images, videos, audio, files, or web content, be specific and helpful
- Reference visual, audio, or file content naturally in conversation

ACTION CAPABILITIES:
You can perform real-world actions! When the user asks, you can:
- WEB SEARCH: Look up anything online
- PLAY MUSIC: Play videos/music from YouTube
- SEND EMAIL: Send emails (if configured)
- OPEN WEBSITES: Open any website
- GET WEATHER: Check weather for any city
- GENERATE IMAGES: Create AI images from descriptions using DALL-E
- GENERATE VIDEOS: Create AI videos from descriptions using AnimateDiff
- READ FILES: Read and analyze any supported file type
- READ WEBPAGES: Extract and read content from any URL
- VIEW IMAGES: See and describe images
- WATCH VIDEOS: Analyze video content and transcribe audio
- LISTEN TO AUDIO: Transcribe audio files
- When you take an action, announce it naturally

CODING CAPABILITIES:
- Expert programmer in ALL languages
- Can create complete, production-ready code
- Can build: web apps, APIs, games, scripts, automation, etc.
- Always provide working code with error handling
- Include helpful comments and best practices"""

    def __init__(self):
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=Config.AWS_REGION
            )
            
            # Test AWS connection
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            print(f"✅ Connected to AWS Bedrock")
            print(f"  Account: {identity['Account']}")
            print(f"  Region: {Config.AWS_REGION}")
            
        except Exception as e:
            print(f"❌ AWS Bedrock connection failed: {e}")
            print("\nPlease run: aws configure")
            print("And enter your AWS credentials")
            raise

    def chat(self, user_message, file_path=None, url=None):
        """Send message to Claude via AWS Bedrock with optional file/image/video/audio/webpage"""
        global conversation_history
        
        # Handle webpage reading if URL provided
        if url:
            success, webpage_content = FileHandler.read_webpage(url)
            if not success:
                return f"Sorry, I couldn't read that webpage: {webpage_content}"
            file_content = webpage_content
        else:
            file_content = None
        
        # Handle file reading if provided
        image_data = None
        video_frames = []
        
        if file_path:
            success, file_type, content = FileHandler.read_file(file_path)
            
            if not success:
                return f"Sorry, I couldn't read that file: {content}"
            
            if file_type == 'image':
                image_data = content
                print(f"📸 Processing image: {image_data['filename']}")
            elif file_type == 'video':
                video_frames = content['frames']
                file_content = f"Video Analysis:\n"
                file_content += f"Duration: {content['duration']:.1f} seconds\n"
                file_content += f"FPS: {content['fps']:.1f}\n"
                file_content += f"Total Frames: {content['total_frames']}\n"
                file_content += f"Extracted {len(video_frames)} key frames for analysis\n"
                if content['audio_transcript']:
                    file_content += f"\nAudio Transcript:\n{content['audio_transcript']}"
                else:
                    file_content += "\nNo audio detected or unable to transcribe"
                print(f"🎬 Processing video with {len(video_frames)} frames")
            elif file_type == 'audio':
                file_content = f"Audio Transcript:\n{content}"
                print(f"🎵 Processed audio file")
            elif file_type == 'text':
                file_content = content
                print(f"📄 Read file content ({len(content)} characters)")
        
        # Handle actions
        action_result = self._handle_actions(user_message)
        
        # Prepare messages
        messages = []
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
        
        # Build current message content
        content_blocks = []
        
        # Add video frames if present
        if video_frames:
            for i, frame_base64 in enumerate(video_frames):
                content_blocks.append({
                    "image": {
                        "format": "jpeg",
                        "source": {
                            "bytes": base64.b64decode(frame_base64)
                        }
                    }
                })
        
        # Add image if present (and no video)
        elif image_data:
            content_blocks.append({
                "image": {
                    "format": image_data['mime_type'].split('/')[-1],
                    "source": {
                        "bytes": base64.b64decode(image_data['data'])
                    }
                }
            })
        
        # Add text message
        message_text = user_message
        if action_result:
            message_text += f"\n[Action Result: {action_result}]"
        if file_content:
            message_text += f"\n\n[File Content]:\n{file_content[:5000]}"  # Limit to 5000 chars
        
        content_blocks.append({"text": message_text})
        
        messages.append({
            "role": "user",
            "content": content_blocks
        })
        
        # Add context
        context_summary = ConversationManager.get_context_summary()
        system_prompt = self.SYSTEM_PROMPT
        if context_summary != "No active projects.":
            system_prompt += f"\n\nCURRENT CONTEXT:\n{context_summary}"
        
        try:
            response = self.bedrock.converse(
            modelId=Config.BEDROCK_MODEL,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={
            "maxTokens": Config.MAX_TOKENS,
            "temperature": 0.7
            }
        )
            
            assistant_response = response['output']['message']['content'][0]['text']
            
            # Save code if present
            if '```' in assistant_response:
                self._save_code_from_response(assistant_response)
            
            # Update history (without image data to save space)
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            ConversationManager.trim_history()
            ConversationManager.save_memory()
            
            return assistant_response
            
        except ClientError as e:
            print(f"Bedrock API Error: {e}")
            return "Sorry, I'm having trouble connecting to AWS right now."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, I had trouble processing that."
    
    def _handle_actions(self, user_message):
        """Detect and execute actions"""
        msg_lower = user_message.lower()
        
        # Animate image (talking with audio or text)
        if any(keyword in msg_lower for keyword in ['animate me', 'make me talk', 'animate my photo', 'make photo talk', 'animate picture', 'lip sync']):
            # Extract image and audio files from the message
            words = user_message.split()
            image_path = None
            audio_path = None
            
            # Look for files
            for word in words:
                clean_word = word.strip('.,!?;:\'"')
                if Path(clean_word).exists():
                    ext = Path(clean_word).suffix.lower()
                    if ext in FileHandler.IMAGE_EXTENSIONS:
                        image_path = clean_word
                    elif ext in FileHandler.AUDIO_EXTENSIONS:
                        audio_path = clean_word
            
            # If no files found, ask for them
            if not image_path:
                return "I need an image file to animate. Please provide: 'animate me [image.jpg] with [audio.mp3]'"
            
            # Check for text to speak if no audio file
            text_to_speak = None
            if not audio_path:
                if 'say' in msg_lower or 'saying' in msg_lower:
                    parts = user_message.lower().split('say')
                    if len(parts) > 1:
                        text_to_speak = parts[1].strip().strip('"\'')
                        # Remove filenames from text
                        for word in words:
                            if '.' in word:
                                text_to_speak = text_to_speak.replace(word, '').strip()
            
            success, result = ActionHandler.animate_image_talking(image_path, audio_path, text_to_speak)
            if success:
                return f"Lip-synced video created: {result}"
            else:
                return f"Animation failed: {result}"
        
        # Animate image (general motion)
        if any(keyword in msg_lower for keyword in ['animate image', 'add motion to']):
            file_path = extract_file_path(user_message)
            if not file_path:
                return "I need an image file to animate. Please provide the image path."
            
            success, result = ActionHandler.animate_image(file_path)
            if success:
                return f"Animated video created: {result}"
            else:
                return f"Animation failed: {result}"
        
        # Video generation
        if any(keyword in msg_lower for keyword in ['generate video', 'create video', 'make video']):
            prompt = user_message
            for keyword in ['generate video of', 'create video of', 'make video of']:
                prompt = prompt.lower().replace(keyword, '').strip()
            
            success, result = ActionHandler.generate_video(prompt)
            if success:
                return f"Video generated: {result}"
            else:
                return f"Video generation failed: {result}"
        
        # Image generation
        if any(keyword in msg_lower for keyword in ['generate image', 'create image', 'make image', 'draw', 'picture of']):
            prompt = user_message
            for keyword in ['generate image of', 'create image of', 'make image of', 'draw', 'picture of']:
                prompt = prompt.lower().replace(keyword, '').strip()
            
            success, result = ActionHandler.generate_image(prompt, "openai")
            if success:
                return f"Image generated: {result}"
            else:
                return f"Image generation failed: {result}"
        
        # Web search
        if any(keyword in msg_lower for keyword in ['search for', 'look up', 'find information', 'google']):
            query = user_message
            for keyword in ['search for', 'look up', 'find information about', 'google']:
                query = query.lower().replace(keyword, '').strip()
            results = ActionHandler.web_search(query, num_results=3)
            if results:
                ActionHandler.open_website(results[0])
                other = ', '.join(results[1:3]) if len(results) > 1 else 'none'
                return f"Opening: {results[0]}. Other results: {other}"
        
        # Read webpage
        if any(keyword in msg_lower for keyword in ['read page', 'read webpage', 'read website', 'read this page', 'what does this page say']):
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user_message)
            if urls:
                success, content = FileHandler.read_webpage(urls[0])
                if success:
                    return f"Webpage read successfully. Content added to context."
                else:
                    return f"Failed to read webpage: {content}"
        
        # YouTube
        youtube_triggers = [
            'play on youtube', 'play youtube', 'youtube', 
            'play music', 'play song', 'play video of'
        ]
        if any(trigger in msg_lower for trigger in youtube_triggers):
            query = user_message
            for keyword in ['play on youtube', 'play youtube', 'on youtube', 'play', 'music', 'video of', 'song']:
                query = query.lower().replace(keyword, '').strip()
            success, url = ActionHandler.play_youtube(query)
            if success:
                return f"Playing: {url}"
        
        # Weather
        if 'weather' in msg_lower:
            words = user_message.split()
            city = None
            if 'in' in msg_lower:
                idx = [i for i, w in enumerate(words) if w.lower() == 'in']
                if idx and idx[0] + 1 < len(words):
                    city = ' '.join(words[idx[0]+1:])
            
            if city:
                weather = ActionHandler.get_weather(city)
                if weather:
                    return weather
        
        # Open website
        if 'open' in msg_lower and any(d in msg_lower for d in ['.com', '.org', '.net']):
            words = user_message.split()
            for word in words:
                if '.' in word:
                    ActionHandler.open_website(word)
                    return f"Opening {word}"
        
        return None
        
       
    def _save_code_from_response(self, response):
        """Extract and save code blocks"""
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            output_dir = Path('code_output')
            output_dir.mkdir(exist_ok=True)
            
            for i, (language, code) in enumerate(code_blocks, 1):
                ext_map = {
                    'python': '.py', 'javascript': '.js', 'java': '.java',
                    'cpp': '.cpp', 'html': '.html', 'css': '.css'
                }
                
                ext = ext_map.get(language.lower(), '.txt') if language else '.txt'
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"sunny_code_{timestamp}_{i}{ext}"
                
                with open(filename, 'w') as f:
                    f.write(code.strip())
                
                print(f"\n💾 Code saved to: {filename}")


class VoiceManager:
    """Handles text-to-speech with AWS Polly and speech-to-text"""
    
    def __init__(self):
        try:
            # Initialize AWS Polly client
            self.polly = boto3.client(
                'polly',
                region_name=Config.AWS_REGION
            )
            print(f"✅ Connected to AWS Polly")
            
        except Exception as e:
            print(f"❌ AWS Polly connection failed: {e}")
            raise
        
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = Config.ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        self.interrupt_flag = threading.Event()
        self.speaking = threading.Event()
        self.keyboard_listener = None
        
    def on_key_press(self, key):
        """Handle keyboard interruption"""
        try:
            if key == keyboard.Key.space and self.speaking.is_set():
                print("\n🛑 [SPACEBAR] Interruption detected!")
                self.interrupt_flag.set()
                return False
        except AttributeError:
            pass
    
    def start_keyboard_monitor(self):
        """Start monitoring for keyboard interrupts"""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
    
    def stop_keyboard_monitor(self):
        """Stop monitoring keyboard"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()

    def speak(self, text):
        """Convert text to speech using AWS Polly"""
        cleaned_text = self._clean_text_for_speech(text)
        
        self.interrupt_flag.clear()
        self.speaking.set()
        self.start_keyboard_monitor()
        
        interrupted = False
        
        try:
            # Call AWS Polly
            response = self.polly.synthesize_speech(
                Text=cleaned_text,
                OutputFormat='mp3',
                VoiceId=Config.POLLY_VOICE_ID,
                Engine=Config.POLLY_ENGINE
            )
            
            if 'AudioStream' in response:
                # Read audio stream
                audio_data = response['AudioStream'].read()
                
                # Play audio using pygame
                sound = pygame.mixer.Sound(io.BytesIO(audio_data))
                print("🔊 Speaking... (Press SPACEBAR to interrupt)")
                sound.play()
                
                while pygame.mixer.get_busy() and not self.interrupt_flag.is_set():
                    pygame.time.wait(50)
                
                if self.interrupt_flag.is_set():
                    pygame.mixer.stop()
                    interrupted = True
            else:
                print("AWS Polly error: No audio stream in response")
                
        except ClientError as e:
            print(f"AWS Polly error: {e}")
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.speaking.clear()
            self.stop_keyboard_monitor()
        
        return interrupted
    
    def _clean_text_for_speech(self, text):
        """Remove non-speech elements"""
        text = re.sub(r'\*[^*]+\*', '', text)
        text = re.sub(r'\*', '', text)
        text = re.sub(r'```[\w]*\n', '[code block]\n', text)
        text = re.sub(r'```', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def listen(self):
        """Listen for voice input with text fallback"""
        try:
            with sr.Microphone() as source:
                print("\n🎤 Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source, 
                    timeout=Config.LISTEN_TIMEOUT, 
                    phrase_time_limit=Config.PHRASE_TIME_LIMIT
                )
            
            print("⏳ Processing...")
            text = self.recognizer.recognize_google(audio)
            print(f"✅ You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️  Listening timed out - Type your message instead:")
            text_input = input("You: ").strip()
            return text_input if text_input else None
        except sr.UnknownValueError:
            print("❌ Could not understand audio - Type your message instead:")
            text_input = input("You: ").strip()
            return text_input if text_input else None
        except Exception as e:
            print(f"❌ Error: {e} - Type your message instead:")
            text_input = input("You: ").strip()
            return text_input if text_input else None


def handle_file_command(user_input, voice, claude):
    """Handle all file-related commands"""
    
    # Detect command type
    command_type = None
    if any(kw in user_input.lower() for kw in ['read file', 'open file', 'analyze file', 'read document']):
        command_type = 'file'
    elif any(kw in user_input.lower() for kw in ['look at', 'see', 'view image', 'analyze image', 'show me']):
        command_type = 'image'
    elif any(kw in user_input.lower() for kw in ['watch', 'analyze video', 'look at video']):
        command_type = 'video'
    elif any(kw in user_input.lower() for kw in ['listen to', 'transcribe', 'hear audio', 'what does', 'play audio']):
        command_type = 'audio'
    elif any(kw in user_input.lower() for kw in ['read page', 'read website', 'summarize page']):
        command_type = 'webpage'
    
    # Also detect if any audio/video/image file extension is mentioned
    if not command_type:
        if any(ext in user_input.lower() for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']):
            command_type = 'audio'
        elif any(ext in user_input.lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
            command_type = 'video'
        elif any(ext in user_input.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
            command_type = 'image'
    
    if not command_type:
        return None, None
    
    # Handle webpage separately
    if command_type == 'webpage':
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user_input)
        if urls:
            return None, urls[0]
        else:
            print("\n🔍 Please enter the URL:")
            url = input("URL: ").strip()
            return None, url if url else None
    
    # Extract file path
    file_path = extract_file_path(user_input)
    
    if not file_path:
        response = "I couldn't find that file. Please make sure the file path is correct."
        print(f"\nSunny: {response}")
        voice.speak(response)
        return None, None
    
    # Verify file exists
    if not Path(file_path).exists():
        response = f"That file doesn't exist: {file_path}"
        print(f"\nSunny: {response}")
        voice.speak(response)
        return None, None
    
    # Verify file type matches command (but be lenient)
    ext = Path(file_path).suffix.lower()
    
    if command_type == 'image' and ext not in FileHandler.IMAGE_EXTENSIONS:
        response = f"That doesn't look like an image file. I found: {file_path}"
        print(f"\nSunny: {response}")
        voice.speak(response)
        return None, None
    
    if command_type == 'video' and ext not in FileHandler.VIDEO_EXTENSIONS:
        response = f"That doesn't look like a video file. I found: {file_path}"
        print(f"\nSunny: {response}")
        voice.speak(response)
        return None, None
    
    if command_type == 'audio' and ext not in FileHandler.AUDIO_EXTENSIONS:
        response = f"That doesn't look like an audio file. I found: {file_path}"
        print(f"\nSunny: {response}")
        voice.speak(response)
        return None, None
    
    print(f"✅ Found {command_type} file: {file_path}")
    return file_path, None

    print("  - Say 'animate me saying [TEXT]' for custom speech")

def main():
    """Main program loop"""
    print("  - Say 'animate me [IMAGE]' to make a photo talk")
    print("  - Say 'animate me saying [TEXT]' for custom speech")
    print("=" * 60)
    print("☀️  SUNNY VOICE ASSISTANT (AWS Edition)")
    print("=" * 60)
    print(f"\n📂 Current directory: {Path.cwd()}")
    print("   (Place files here or provide full path)")
    print("\nCommands:")
    print("  - Say 'quit', 'exit', or 'stop' to end")
    print("  - Say 'clear memory' to reset conversation")
    print("  - Say 'list files' to see files in current directory")
    print("  - Say 'generate image of X' to create images")
    print("  - Say 'generate video of X' to create videos")
    print("  - Say 'read file FILENAME' to read any file")
    print("  - Say 'read page URL' to read any webpage")
    print("  - Say 'look at IMAGE' to analyze images")
    print("  - Say 'watch VIDEO' to analyze videos")
    print("  - Say 'listen to AUDIO' to transcribe audio")
    print("  - Say 'play music/song NAME' to play on YouTube")
    print("  - Press SPACEBAR to interrupt Sunny")
    print("  - Press Ctrl-C to exit anytime")
    print("=" * 60)
    
    try:
        ConversationManager.load_memory()
        ConversationManager.load_projects()
        claude = BedrockClient()
        voice = VoiceManager()
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        return
    
    EXIT_COMMANDS = {'quit', 'exit', 'stop', 'goodbye'}
    
    while True:
        try:
            user_input = voice.listen()
            
            if user_input is None:
                continue
            
            # Check for exit commands
            if user_input.lower() in EXIT_COMMANDS:
                response = "Goodbye! Have a great day!"
                print(f"\nSunny: {response}")
                voice.speak(response)
                break
            
            # Check for clear memory command
            if 'clear memory' in user_input.lower():
                ConversationManager.clear_memory()
                response = "Memory cleared! Starting fresh."
                print(f"\nSunny: {response}")
                voice.speak(response)
                continue
            
            # Check for list files command
            if 'list files' in user_input.lower() or 'show files' in user_input.lower():
                files = list(Path.cwd().glob('*.*'))
                file_list = ', '.join([f.name for f in files[:20]])
                response = f"Files in current directory: {file_list}"
                if len(files) > 20:
                    response += f"... and {len(files) - 20} more"
                print(f"\nSunny: {response}")
                voice.speak(response)
                continue
            
            # Check for file/image/video/audio/webpage commands
            file_path = None
            url = None
            
            if any(keyword in user_input.lower() for keyword in 
                   ['read file', 'open file', 'look at', 'see', 'view', 
                    'watch', 'listen to', 'transcribe', 'read page', 
                    'read website', 'analyze']):
                file_path, url = handle_file_command(user_input, voice, claude)
                if file_path is None and url is None:
                    continue  # Error already handled
            
            # Get response from Claude
            response = claude.chat(user_input, file_path=file_path, url=url)
            print(f"\nSunny: {response}")
            
            # Speak response
            interrupted = voice.speak(response)
            
            if interrupted:
                print("\n💬 What did you need?")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()