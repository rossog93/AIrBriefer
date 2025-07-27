import os
from src.ai_processor import AITranscriber

from src.utils.logger import logger
logger.info("initializing AITranscriber")

audio_file = os.path.join(os.getcwd(), 'test', 'audio_test.ogg')
transcriber = AITranscriber(model_name="large")  # Use your desired Whisper model
transcription = transcriber.transcribe(audio_file, reduce_noise=True) #add a reduce noise flag

print(transcription)
