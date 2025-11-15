import whisper
import pyaudio
import wave
import os
import threading
import time
from pydub import AudioSegment
import re
import json
from datetime import datetime
import soundfile as sf
import sys

class AudioNoteTaker:
    def __init__(self):
        self.model = None
        self.is_recording = False
        self.audio_frames = []
        self.notes_file = "audio_notes.json"
        self.audio = None
        self.load_model()
        
    def load_model(self):
        """Load Whisper model"""
        print("Loading Whisper model...")
        try:
            self.model = whisper.load_model("base")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def list_audio_devices(self):
        """List available audio devices"""
        try:
            self.audio = pyaudio.PyAudio()
            print("\nAvailable audio devices:")
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"Device {i}: {device_info['name']} (Input Channels: {device_info['maxInputChannels']})")
            return True
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            return False

    def test_audio_input(self):
        """Test if audio input is working"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Test recording for 2 seconds
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("Testing audio input... Speak something for 2 seconds")
            frames = []
            
            for i in range(0, int(RATE / CHUNK * 2)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Save test recording
            if frames:
                test_file = "test_audio.wav"
                wf = wave.open(test_file, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Check file size to see if audio was recorded
                file_size = os.path.getsize(test_file)
                if file_size > 1000:  # If file is larger than 1KB, audio was recorded
                    print(f"✅ Audio test successful! Recorded {file_size} bytes")
                    # Try to transcribe
                    text = self.transcribe_audio(test_file)
                    if text:
                        print(f"Test transcription: '{text}'")
                    os.remove(test_file)
                    return True
                else:
                    print("❌ No audio detected - file too small")
                    os.remove(test_file)
                    return False
            else:
                print("❌ No audio frames recorded")
                return False
                
        except Exception as e:
            print(f"❌ Audio test failed: {e}")
            return False

    def record_audio(self, duration=10, filename="temp_audio.wav"):
        """Record audio for specified duration"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100  # Increased sample rate for better quality
            
            if not self.audio:
                self.audio = pyaudio.PyAudio()
            
            print(f"🎤 Recording for {duration} seconds...")
            
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            
            for i in range(0, int(RATE / CHUNK * duration)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    # Show recording progress
                    if i % (RATE // CHUNK) == 0:  # Every second
                        seconds_recorded = i // (RATE // CHUNK)
                        print(f"⏺️  Recording... {seconds_recorded}/{duration}s")
                except Exception as e:
                    print(f"Error during recording: {e}")
                    break
            
            print("✅ Recording finished")
            
            stream.stop_stream()
            stream.close()
            
            # Save recorded audio
            if frames:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Verify recording
                file_size = os.path.getsize(filename)
                print(f"📁 Audio saved: {filename} ({file_size} bytes)")
                
                return filename
            else:
                print("❌ No audio frames recorded")
                return None
                
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None

    def continuous_recording(self, stop_event, filename="continuous_audio.wav"):
        """Continuous recording for command detection"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            
            if not self.audio:
                self.audio = pyaudio.PyAudio()
                
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            print("🎤 Continuous recording started... Say 'stop' to end.")
            print("Press Ctrl+C to stop manually")
            
            last_command_check = time.time()
            
            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Check for commands every 3 seconds
                    current_time = time.time()
                    if current_time - last_command_check >= 3:
                        if self.process_recent_audio(frames, RATE, CHUNK):
                            print("🛑 Stop command detected!")
                            break
                        last_command_check = current_time
                        
                except Exception as e:
                    print(f"Error in continuous recording: {e}")
                    break
            
            # Save final audio
            if frames:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print(f"📁 Continuous audio saved: {filename}")
            
            stream.stop_stream()
            stream.close()
            return filename
            
        except Exception as e:
            print(f"❌ Continuous recording error: {e}")
            return None

    def process_recent_audio(self, frames, rate, chunk_size):
        """Process recent audio for commands"""
        try:
            # Use last 2 seconds of audio
            samples_per_second = rate // chunk_size
            recent_duration = 2  # seconds
            recent_frames_count = recent_duration * samples_per_second
            
            if len(frames) < recent_frames_count:
                return False
                
            recent_frames = frames[-recent_frames_count:]
            temp_filename = "temp_command.wav"
            
            # Save recent audio
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(recent_frames))
            wf.close()
            
            # Check for commands
            if self.check_for_commands(temp_filename):
                return True
                
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            return False
            
        except Exception as e:
            print(f"Command processing error: {e}")
            return False

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper"""
        try:
            if not os.path.exists(audio_file):
                print(f"❌ Audio file not found: {audio_file}")
                return ""
                
            # Check file size
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                print(f"❌ Audio file too small: {file_size} bytes")
                return ""
                
            print("🔍 Transcribing audio...")
            result = self.model.transcribe(audio_file)
            text = result["text"].strip()
            
            if text:
                print(f"📝 Transcription: '{text}'")
            else:
                print("❌ No speech detected in audio")
                
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def clean_text(self, text):
        """Clean and format transcribed text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        # Add period if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
            
        return text

    def check_for_commands(self, audio_file):
        """Check audio for specific commands"""
        try:
            text = self.transcribe_audio(audio_file)
            text_lower = text.lower()
            
            commands = {
                'stop': self.stop_recording_command,
                'example': self.insert_example_command,
                'math': self.math_example_command,
                'new section': self.new_section_command,
            }
            
            for keyword, command_func in commands.items():
                if keyword in text_lower:
                    print(f"🎯 Command detected: '{keyword}'")
                    command_func(text)
                    return True
            return False
            
        except Exception as e:
            print(f"Command check error: {e}")
            return False

    def extract_mathematical_content(self, text):
        """Extract mathematical expressions and problems"""
        math_patterns = [
            r'\b\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+\b',
            r'\b\d+\s*[\+\-\*\/]\s*\d+\b',
            r'\bsolve\s+.*?\b',
            r'\bcalculate\s+.*?\b',
        ]
        
        math_content = []
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            math_content.extend(matches)
            
        return math_content

    def format_mathematical_example(self, math_content):
        """Format mathematical examples for notes"""
        formatted_examples = []
        for example in math_content:
            formatted_example = f"**Mathematical Example:**\n"
            formatted_example += f"Problem: {example}\n"
            
            # Simple evaluation for basic math
            try:
                clean_expr = re.sub(r'(solve|calculate)\s+', '', example, flags=re.IGNORECASE)
                if re.match(r'^[\d\s\.\+\-\*\/]+$', clean_expr):
                    result = eval(clean_expr)
                    formatted_example += f"Solution: {result}\n"
            except:
                formatted_example += f"Solution: [Work through solution]\n"
                
            formatted_examples.append(formatted_example)
            
        return formatted_examples

    # Command handlers
    def stop_recording_command(self, text):
        """Handle stop recording command"""
        self.is_recording = False
        print("🛑 Stop command received!")

    def insert_example_command(self, text):
        """Handle insert example command"""
        print("💡 Example command detected!")
        self.add_example_to_notes("general", text)

    def new_section_command(self, text):
        """Handle new section command"""
        section_name = "New Section"
        if 'new section' in text.lower():
            parts = text.lower().split('new section')
            if len(parts) > 1 and parts[1].strip():
                section_name = parts[1].strip().title()
        
        print(f"📁 Creating new section: {section_name}")
        self.add_section_to_notes(section_name)

    def math_example_command(self, text):
        """Handle mathematical example command"""
        print("🔢 Math example command detected!")
        math_content = self.extract_mathematical_content(text)
        if math_content:
            formatted_examples = self.format_mathematical_example(math_content)
            for example in formatted_examples:
                self.add_example_to_notes("mathematical", example)

    # Notes management
    def load_notes(self):
        """Load existing notes from file"""
        try:
            with open(self.notes_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"sections": [{"name": "Main", "content": []}], "examples": []}

    def save_notes(self, notes):
        """Save notes to file"""
        try:
            with open(self.notes_file, 'w') as f:
                json.dump(notes, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving notes: {e}")

    def add_note(self, text, section="Main"):
        """Add a new note"""
        if not text.strip():
            print("❌ Empty text - note not added")
            return
            
        notes = self.load_notes()
        
        # Find or create section
        section_exists = False
        for sec in notes["sections"]:
            if sec["name"].lower() == section.lower():
                sec["content"].append({
                    "timestamp": datetime.now().isoformat(),
                    "text": text
                })
                section_exists = True
                break
        
        if not section_exists:
            notes["sections"].append({
                "name": section,
                "content": [{
                    "timestamp": datetime.now().isoformat(),
                    "text": text
                }]
            })
        
        self.save_notes(notes)
        print(f"✅ Note added to section: {section}")

    def add_example_to_notes(self, example_type, content):
        """Add example to notes"""
        notes = self.load_notes()
        
        notes["examples"].append({
            "type": example_type,
            "timestamp": datetime.now().isoformat(),
            "content": content
        })
        
        self.save_notes(notes)
        print(f"✅ {example_type.capitalize()} example added!")

    def add_section_to_notes(self, section_name):
        """Add new section to notes"""
        notes = self.load_notes()
        
        # Check if section already exists
        for section in notes["sections"]:
            if section["name"].lower() == section_name.lower():
                print(f"❌ Section '{section_name}' already exists!")
                return
        
        notes["sections"].append({
            "name": section_name,
            "content": []
        })
        
        self.save_notes(notes)
        print(f"✅ New section '{section_name}' created!")

    # Main application methods
    def start_note_taking_session(self):
        """Main function to start note taking session"""
        print("🎙️  Audio Note Taker")
        print("=" * 50)
        
        # Test audio system first
        print("\n1. Testing audio system...")
        if not self.test_audio_input():
            print("\n❌ Audio system test failed!")
            print("Troubleshooting steps:")
            print("1. Check microphone permissions")
            print("2. Ensure microphone is not muted")
            print("3. Try different audio device")
            self.list_audio_devices()
            return
        
        print("\n✅ Audio system ready!")
        print("\n🎯 Voice Commands:")
        print("- 'stop' - End recording")
        print("- 'example' - Insert an example") 
        print("- 'math' - Insert mathematical example")
        print("- 'new section [name]' - Create new section")
        print("=" * 50)
        
        while True:
            print("\n📋 Options:")
            print("1. Record specific duration")
            print("2. Continuous recording (voice commands)")
            print("3. View notes")
            print("4. Export notes to text")
            print("5. Test audio again")
            print("6. Exit")
            
            choice = input("Choose option (1-6): ").strip()
            
            if choice == "1":
                self.record_specific_duration()
            elif choice == "2":
                self.continuous_note_taking()
            elif choice == "3":
                self.view_notes()
            elif choice == "4":
                self.export_notes_to_txt()
            elif choice == "5":
                self.test_audio_input()
            elif choice == "6":
                print("👋 Goodbye!")
                if self.audio:
                    self.audio.terminate()
                break
            else:
                print("❌ Invalid choice!")

    def record_specific_duration(self):
        """Record for specific duration"""
        try:
            duration = int(input("⏱️  Recording duration (seconds): "))
            section = input("📁 Section name (press enter for 'Main'): ").strip()
            if not section:
                section = "Main"
                
            audio_file = self.record_audio(duration)
            if audio_file and os.path.exists(audio_file):
                text = self.transcribe_audio(audio_file)
                cleaned_text = self.clean_text(text)
                
                if cleaned_text:
                    # Check for mathematical content
                    math_content = self.extract_mathematical_content(text)
                    if math_content:
                        print(f"🔢 Math content found!")
                        for example in self.format_mathematical_example(math_content):
                            self.add_example_to_notes("mathematical", example)
                    
                    # Add to notes
                    self.add_note(cleaned_text, section)
                else:
                    print("❌ No text transcribed - note not added")
                
                # Clean up
                os.remove(audio_file)
            else:
                print("❌ Recording failed!")
                
        except ValueError:
            print("❌ Please enter a valid number!")
        except Exception as e:
            print(f"❌ Error: {e}")

    def continuous_note_taking(self):
        """Continuous note taking with voice commands"""
        print("\n🎤 Continuous Recording Mode")
        print("Say 'stop' to end recording...")
        
        stop_event = threading.Event()
        self.is_recording = True
        
        # Start recording in separate thread
        recording_thread = threading.Thread(
            target=self.continuous_recording, 
            args=(stop_event, "continuous_session.wav")
        )
        recording_thread.daemon = True
        recording_thread.start()
        
        try:
            # Wait for stop command or manual input
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Manual stop requested...")
            self.is_recording = False
        
        stop_event.set()
        recording_thread.join(timeout=5)
        
        # Process final recording
        if os.path.exists("continuous_session.wav"):
            text = self.transcribe_audio("continuous_session.wav")
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                print(f"\n📝 Final transcription: {cleaned_text}")
                self.add_note(cleaned_text, "Continuous Session")
            else:
                print("❌ No text transcribed from continuous session")
            
            # Clean up
            os.remove("continuous_session.wav")
        else:
            print("❌ No continuous session recording found")

    def view_notes(self):
        """Display all notes"""
        notes = self.load_notes()
        
        print("\n📓 YOUR NOTES")
        print("=" * 50)
        
        for section in notes["sections"]:
            print(f"\n📁 {section['name'].upper()}")
            print("-" * 30)
            if section["content"]:
                for note in section["content"]:
                    timestamp = datetime.fromisoformat(note["timestamp"]).strftime("%m/%d %H:%M")
                    print(f"🕒 {timestamp}: {note['text']}")
            else:
                print("No notes in this section")
        
        if notes["examples"]:
            print(f"\n💡 EXAMPLES")
            print("-" * 30)
            for example in notes["examples"]:
                timestamp = datetime.fromisoformat(example["timestamp"]).strftime("%m/%d %H:%M")
                print(f"🕒 {timestamp} [{example['type']}]:")
                print(f"   {example['content']}\n")

    def export_notes_to_txt(self, filename="notes_export.txt"):
        """Export notes to text file"""
        notes = self.load_notes()
        
        try:
            with open(filename, 'w') as f:
                f.write("AUDIO NOTES EXPORT\n")
                f.write("=" * 50 + "\n\n")
                
                for section in notes["sections"]:
                    f.write(f"SECTION: {section['name']}\n")
                    f.write("-" * 30 + "\n")
                    
                    for note in section["content"]:
                        timestamp = datetime.fromisoformat(note["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] {note['text']}\n")
                    f.write("\n")
                
                if notes["examples"]:
                    f.write("EXAMPLES\n")
                    f.write("-" * 30 + "\n")
                    for example in notes["examples"]:
                        timestamp = datetime.fromisoformat(example["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] {example['type'].upper()}:\n")
                        f.write(f"{example['content']}\n\n")
            
            print(f"✅ Notes exported to {filename}")
        except Exception as e:
            print(f"❌ Export error: {e}")

def main():
    """Main function to run the application"""
    try:
        note_taker = AudioNoteTaker()
        note_taker.start_note_taking_session()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'note_taker' in locals() and note_taker.audio:
            note_taker.audio.terminate()

if __name__ == "__main__":
    main()