!pip install torch

import webrtcvad
from pydub import AudioSegment
from faster_whisper import WhisperModel
import IPython.display as ipd
from edge_tts import Communicate
from transformers import AutoTokenizer, AutoModelForCausalLM
import edge_tts
import IPython.display as ipd

def apply_vad(audio_file, vad_threshold=0.5):
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Ensure mono and 16kHz
    
    # Convert audio to raw data for VAD processing
    raw_data = audio.raw_data
    vad = webrtcvad.Vad(int(vad_threshold * 3))  # VAD aggressiveness: 0, 1, 2, 3
    
    frame_duration = 30  # 30 ms
    frame_size = int(16000 * (frame_duration / 1000.0) * 2)  # 30ms frame
    segments = bytearray()

    for i in range(0, len(raw_data), frame_size):
        frame = raw_data[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate=16000):
            segments.extend(frame)
    
    # Recreate the audio segment with the VAD-applied data
    vad_audio = AudioSegment(
        data=bytes(segments),
        sample_width=audio.sample_width,
        frame_rate=audio.frame_rate,
        channels=audio.channels
    )
    
    return vad_audio

def speech_to_text_whisper(audio_file):
    # Apply VAD to the audio file
    vad_audio = apply_vad(audio_file, vad_threshold=0.5)
    
    # Save the VAD-processed audio to a temporary file
    vad_audio_path = "vad_audio.wav"
    vad_audio.export(vad_audio_path, format="wav")
    
    # Load the Whisper model with float32 as the compute type, use the cpu instead of the cuda
    model = WhisperModel("medium.en", device="cpu", compute_type="float32") #Modified this line to use the cpu
    
    # Transcribe the VAD-processed audio
    segments, info = model.transcribe(vad_audio_path, beam_size=5)
    
    # Combine the segments into a single string
    text = ' '.join([segment.text for segment in segments])
    return text

def generate_text_with_llm(input_text):
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    output = model.generate(**inputs, max_new_tokens=150)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Limit the output to 9 sentences
    sentences = decoded_output.split('. ')
    limited_output = '.'.join(sentences[:6]) + '.'

    return limited_output

async def text_to_speech_tunable(text, voice="en-US-JennyNeural", rate="+0%", pitch="+0Hz"):
    # Initialize the edge-tts Communicate object
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    
    # Synthesize and save the output audio
    await communicate.save("output_audio.mp3")

    # Play the audio file in a Jupyter notebook or similar environment
    ipd.display(ipd.Audio("output_audio.mp3"))

audio_path = "/content/A1.wav"

# Step 1: Convert speech to text
text = speech_to_text_whisper(audio_path)
print("Transcribed Text:", text)
