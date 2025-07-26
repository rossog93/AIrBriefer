import os
from src.ai_processor import AITranscriber


audio_file = os.path.join(os.getcwd(), 'test', 'audio_test.ogg')
transcriber = AITranscriber(model_name="base")  # Use your desired Whisper model
transcription = transcriber.transcribe(audio_file)
print(transcription)