import streamlit as st
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
from openai import OpenAI
import tempfile
import queue

# Initialize session state
def init_session_state():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_frames' not in st.session_state:
        st.session_state.audio_frames = []
    if 'notes' not in st.session_state:
        st.session_state.notes = {}
    if 'current_file' not in st.session_state:
        st.session_state.current_file = "default_notes.txt"
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if 'llm_responses' not in st.session_state:
        st.session_state.llm_responses = []
    if 'recording_thread' not in st.session_state:
        st.session_state.recording_thread = None
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()

class AudioNoteTaker:
    def __init__(self):
        self.model = None
        self.audio = None
        self.load_model()
        
    def load_model(self):
        """Load Whisper model"""
        try:
            with st.spinner("Loading Whisper model..."):
                self.model = whisper.load_model("base")
            st.success("Whisper model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    def initialize_audio(self):
        """Initialize audio system"""
        try:
            self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            st.error(f"Audio initialization failed: {e}")
            return False

    def record_audio(self, stop_event, audio_frames):
        """Record audio in a thread-safe way"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100

            if not self.audio:
                if not self.initialize_audio():
                    return

            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Clear previous frames
            audio_frames.clear()
            
            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_frames.append(data)
                    
                    # Check for commands every 2 seconds
                    if len(audio_frames) % (2 * RATE // CHUNK) == 0:
                        self.process_realtime_audio(audio_frames, RATE, CHUNK)
                        
                except Exception as e:
                    st.error(f"Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            st.error(f"Recording thread error: {e}")

    def process_realtime_audio(self, frames, rate, chunk_size):
        """Process audio in real-time for Jarvis commands"""
        try:
            # Use last 3 seconds of audio
            samples_per_second = rate // chunk_size
            recent_duration = 3
            recent_frames_count = recent_duration * samples_per_second
            
            if len(frames) < recent_frames_count:
                return
                
            recent_frames = frames[-recent_frames_count:]
            temp_filename = "temp_realtime.wav"
            
            # Save recent audio
            self.save_audio_frames_threadsafe(recent_frames, temp_filename, rate)
            
            # Check for Jarvis commands
            self.detect_jarvis_commands(temp_filename)
            
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        except Exception as e:
            # Don't show error in real-time processing to avoid spam
            pass

    def save_audio_frames_threadsafe(self, frames, filename, rate):
        """Thread-safe audio frame saving"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        except Exception as e:
            pass

    def detect_jarvis_commands(self, audio_file):
        """Detect Jarvis commands in audio"""
        try:
            text = self.transcribe_audio(audio_file)
            if not text:
                return
                
            text_lower = text.lower()
            
            # Look for Jarvis command pattern
            if "jarvis" in text_lower:
                # Simple command detection - look for text after "jarvis"
                start_idx = text_lower.find("jarvis") + len("jarvis")
                command = text[start_idx:].strip()
                
                if command and len(command) > 5:  # Minimum command length
                    # Use a queue or other thread-safe method to communicate with main thread
                    st.session_state.pending_command = command
                    
        except Exception as e:
            pass

    def process_jarvis_command(self, command):
        """Process Jarvis command with LLM"""
        try:
            # Initialize OpenAI client
            api_key = st.session_state.get('openai_api_key', '')
            if not api_key:
                st.warning("Please enter your OpenAI API key in the sidebar")
                return
                
            client = OpenAI(api_key=api_key)
            
            # Create prompt for the LLM
            prompt = f"""
            You are Jarvis, an AI assistant. The user has given you this command: "{command}"
            
            Please provide a helpful, concise response that would be useful to include in notes.
            Focus on being informative and practical.
            Keep your response under 200 words.
            """
            
            with st.spinner("🤖 Jarvis is thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are Jarvis, a helpful AI assistant that provides useful information for note-taking."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Store response
            st.session_state.llm_responses.append({
                "command": command,
                "response": llm_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Add to notes automatically
            note_text = f"🤖 JARVIS: {command}\n{llm_response}\n"
            self.add_to_notes(note_text)
            
            return llm_response
            
        except Exception as e:
            st.error(f"LLM processing error: {e}")
            return None

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper"""
        try:
            if not os.path.exists(audio_file):
                return ""
                
            result = self.model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            return ""

    def add_to_notes(self, text):
        """Add text to current notes file"""
        try:
            filename = st.session_state.current_file
            if filename not in st.session_state.notes:
                st.session_state.notes[filename] = []
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.notes[filename].append({
                "timestamp": timestamp,
                "text": text
            })
            
        except Exception as e:
            st.error(f"Error adding to notes: {e}")

    def save_notes_to_file(self):
        """Save notes to current file"""
        try:
            filename = st.session_state.current_file
            if filename in st.session_state.notes and st.session_state.notes[filename]:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"NOTES - {filename}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for note in st.session_state.notes[filename]:
                        f.write(f"[{note['timestamp']}]\n")
                        f.write(f"{note['text']}\n")
                        f.write("-" * 30 + "\n")
                
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error saving notes: {e}")
            return False

def start_recording():
    """Start recording - called from main thread"""
    init_session_state()
    
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.stop_event.clear()
        st.session_state.audio_frames = []
        
        # Start recording in separate thread
        recording_thread = threading.Thread(
            target=st.session_state.audio_taker.record_audio,
            args=(st.session_state.stop_event, st.session_state.audio_frames)
        )
        recording_thread.daemon = True
        recording_thread.start()
        st.session_state.recording_thread = recording_thread
        
        return True
    return False

def stop_recording():
    """Stop recording and process audio - called from main thread"""
    init_session_state()
    
    if st.session_state.recording:
        st.session_state.recording = False
        st.session_state.stop_event.set()
        
        # Wait for thread to finish
        if st.session_state.recording_thread:
            st.session_state.recording_thread.join(timeout=2)
        
        # Process the recorded audio
        if st.session_state.audio_frames:
            with st.spinner("Processing audio..."):
                # Save the recording
                temp_audio = "temp_recording.wav"
                st.session_state.audio_taker.save_audio_frames_threadsafe(
                    st.session_state.audio_frames, 
                    temp_audio, 
                    44100
                )
                
                # Transcribe
                transcription = st.session_state.audio_taker.transcribe_audio(temp_audio)
                if transcription:
                    st.session_state.transcription = transcription
                    # Add to notes
                    st.session_state.audio_taker.add_to_notes(transcription)
                
                # Clean up
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            
            st.session_state.audio_frames = []
            return True
        
    return False

def main():
    st.set_page_config(
        page_title="Jarvis Audio Notes",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Jarvis Audio Note Taker")
    st.markdown("Record notes and use voice commands starting with **'Jarvis'** to get AI assistance!")
    
    # Initialize session state and audio system
    init_session_state()
    
    if 'audio_taker' not in st.session_state:
        st.session_state.audio_taker = AudioNoteTaker()
    
    # Check for pending commands from background thread
    if hasattr(st.session_state, 'pending_command'):
        command = st.session_state.pending_command
        with st.spinner(f"Processing Jarvis command: {command}"):
            response = st.session_state.audio_taker.process_jarvis_command(command)
            if response:
                st.success("✅ Jarvis response added to notes!")
        # Remove the pending command
        del st.session_state.pending_command
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key for Jarvis commands",
                               value=st.session_state.get('openai_api_key', ''))
        if api_key:
            st.session_state.openai_api_key = api_key
        
        # File management
        st.subheader("📁 File Management")
        
        col1, col2 = st.columns(2)
        with col1:
            new_filename = st.text_input("New file name", value="my_notes.txt", key="new_filename")
        with col2:
            if st.button("Create New File"):
                st.session_state.current_file = new_filename
                if new_filename not in st.session_state.notes:
                    st.session_state.notes[new_filename] = []
                st.success(f"Created new file: {new_filename}")
        
        # File selection
        existing_files = list(st.session_state.notes.keys())
        if existing_files:
            selected_file = st.selectbox("Select file", existing_files, 
                                       index=existing_files.index(st.session_state.current_file) 
                                       if st.session_state.current_file in existing_files else 0,
                                       key="file_selector")
            if selected_file != st.session_state.current_file:
                st.session_state.current_file = selected_file
        
        # Save button
        if st.button("💾 Save Notes to File"):
            if st.session_state.audio_taker.save_notes_to_file():
                st.success("✅ Notes saved successfully!")
            else:
                st.warning("No notes to save")
        
        st.markdown("---")
        st.subheader("🎯 Voice Commands")
        st.markdown("""
        **Say during recording:**
        - `Jarvis [your command]`
        - Example: *"Jarvis explain quantum computing"*
        """)
        
        st.markdown("---")
        st.subheader("🔧 Audio Status")
        if st.session_state.recording:
            st.error("🔴 Recording...")
        else:
            st.success("🟢 Ready to record")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎤 Recording Controls")
        
        # Recording buttons
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if not st.session_state.recording:
                if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                    if start_recording():
                        st.success("🎙️ Recording started! Speak now...")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start recording")
        
        with col_rec2:
            if st.session_state.recording:
                if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                    if stop_recording():
                        st.success("✅ Recording stopped and processed!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop recording")
        
        # Recording instructions
        if st.session_state.recording:
            st.info("""
            **Recording in progress...**
            - Speak clearly into your microphone
            - Use **"Jarvis"** followed by your command for AI assistance
            - Click **Stop Recording** when finished
            """)
        
        # Current transcription
        if st.session_state.transcription:
            st.subheader("📝 Latest Transcription")
            st.text_area("Transcribed Text", st.session_state.transcription, height=150, key="transcription_display")
            
            col_add, col_clear = st.columns(2)
            with col_add:
                if st.button("Add to Notes", key="add_transcription"):
                    st.session_state.audio_taker.add_to_notes(st.session_state.transcription)
                    st.session_state.transcription = ""
                    st.success("✅ Added to notes!")
                    st.rerun()
            with col_clear:
                if st.button("Clear", key="clear_transcription"):
                    st.session_state.transcription = ""
                    st.rerun()
    
    with col2:
        st.header("📝 Current Notes")
        
        # Current file info
        st.info(f"**Current File:** `{st.session_state.current_file}`")
        
        # Display notes
        current_file = st.session_state.current_file
        if current_file in st.session_state.notes and st.session_state.notes[current_file]:
            notes = st.session_state.notes[current_file]
            
            # Show note count
            st.metric("Total Notes", len(notes))
            
            # Display recent notes (last 5)
            st.subheader("Recent Notes:")
            for i, note in enumerate(reversed(notes[-5:])):
                with st.expander(f"Note {len(notes)-i} - {note['timestamp']}"):
                    st.write(note['text'])
                    
                    # Delete button for each note
                    if st.button(f"Delete Note {len(notes)-i}", key=f"delete_{len(notes)-i}"):
                        # Remove the note
                        original_index = len(notes) - 1 - i
                        st.session_state.notes[current_file].pop(original_index)
                        st.success("Note deleted!")
                        st.rerun()
        else:
            st.info("No notes yet. Start recording to add notes!")
    
    # Jarvis responses section
    if st.session_state.llm_responses:
        st.header("🤖 Jarvis Responses")
        for i, response in enumerate(reversed(st.session_state.llm_responses[-3:])):
            with st.expander(f"Jarvis Response {len(st.session_state.llm_responses)-i}"):
                st.write(f"**Command:** {response['command']}")
                st.write(f"**Response:** {response['response']}")
                st.write(f"*{response['timestamp']}*")
    
    # Manual note input
    st.header("✏️ Manual Note Entry")
    manual_note = st.text_area("Type your note here:", key="manual_note", height=100)
    col_manual1, col_manual2 = st.columns(2)
    with col_manual1:
        if st.button("Add Manual Note") and manual_note.strip():
            st.session_state.audio_taker.add_to_notes(manual_note)
            st.success("✅ Manual note added!")
            st.rerun()
    with col_manual2:
        if st.button("Clear Manual Input"):
            st.rerun()
    
    # File preview and download
    st.header("📄 File Operations")
    if current_file in st.session_state.notes and st.session_state.notes[current_file]:
        notes_content = ""
        for note in st.session_state.notes[current_file]:
            notes_content += f"[{note['timestamp']}]\n{note['text']}\n{'-'*40}\n"
        
        st.subheader("File Preview")
        st.text_area("Full Content", notes_content, height=200, key="file_preview")
        
        # Download button
        st.download_button(
            label="📥 Download Notes as TXT",
            data=notes_content,
            file_name=current_file,
            mime="text/plain"
        )
        
        # Clear all notes button
        if st.button("🗑️ Clear All Notes", type="secondary"):
            st.session_state.notes[current_file] = []
            st.success("All notes cleared!")
            st.rerun()
    else:
        st.info("No notes available for download")

if __name__ == "__main__":
    main()