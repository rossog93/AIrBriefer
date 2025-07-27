# src/ai_transcriber.py
import whisper
import logging
import noisereduce as nr
from pydub import AudioSegment
import numpy as np
from src.utils.logger import logger

class AITranscriber:
    def __init__(self, model_name, language="es"):
        self.language = language
        logging.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def reduce_noise(self, input_path):
        logger.info(f"Reducing noise for audio: {input_path}")
        audio = AudioSegment.from_file(input_path)
        samples = np.array(audio.get_array_of_samples())
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
        clean_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        logger.info(f"Noise reduction completed, exporting clean audio.")

        input_path=input_path.split(".ogg")[0]+ "_clean.ogg"
        clean_audio.export(input_path, format="ogg")
        logger.info("Clean audio exported successfully.")

    def transcribe(self, audio_file, reduce_noise=False):
        try:
            logging.info(f"Transcribing audio: {audio_file}")

            if reduce_noise:
                logging.info("Noise reduction enabled, processing clean audio.")
                self.reduce_noise(audio_file)
                result = self.model.transcribe(audio_file)
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            logging.info(f"Transcription completed: {transcription}")
            return transcription

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""
