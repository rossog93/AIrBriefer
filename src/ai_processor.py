# src/ai_transcriber.py
import whisper
import logging
import noisereduce as nr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import logger
import os

class AITranscriber:
    def __init__(self, model_name, language="es"):
        self.language = language
        logging.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def reduce_noise(self, input_path, plt=True):
        logger.info(f"Reducing noise for audio: {input_path}")
        audio = AudioSegment.from_file(input_path)
        samples = np.array(audio.get_array_of_samples())

        # Step 1: Noise reduction
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
        clean_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels)

        # Step 2: Filtering (band-pass)
        logger.info("Applying band-pass filter (300–3400 Hz).")
        filtered_audio = clean_audio.high_pass_filter(300).low_pass_filter(3400)
        filtered_samples = np.array(filtered_audio.get_array_of_samples())

        # Step 3: Normalization
        logger.info("Normalizing audio amplitude.")
        normalized_audio = filtered_audio.normalize()
        final_samples = np.array(normalized_audio.get_array_of_samples())

        logger.info("Noise reduction, filtering, and normalization completed. Exporting clean audio.")
        output_path = input_path.split(".ogg")[0] + "_clean.ogg"
        normalized_audio.export(output_path, format="ogg")
        logger.info("Clean audio exported successfully.")

        if plt:
            self.plot_signals(
                original=samples,
                reduced=reduced_noise,
                filtered=filtered_samples,
                final=final_samples,
                sample_rate=audio.frame_rate)

    def plot_signals(self, original, reduced, filtered, final, sample_rate):
        def compute_fft(signal):
            freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
            fft_vals = np.abs(np.fft.rfft(signal))
            return freqs, fft_vals

        freqs_orig, fft_orig = compute_fft(original)
        freqs_red, fft_red = compute_fft(reduced)
        freqs_fil, fft_fil = compute_fft(filtered)
        freqs_fin, fft_fin = compute_fft(final)

        plt.figure(figsize=(20, 12))

        # Time domain
        plt.subplot(2, 4, 1)
        plt.plot(original)
        plt.title("Original (Time)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 4, 2)
        plt.plot(reduced, color='orange')
        plt.title("After Noise Reduction (Time)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 4, 3)
        plt.plot(filtered, color='purple')
        plt.title("After Filtering (Time)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 4, 4)
        plt.plot(final, color='green')
        plt.title("After Normalization (Time)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Frequency domain
        plt.subplot(2, 4, 5)
        plt.plot(freqs_orig, fft_orig)
        plt.title("Original (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        plt.subplot(2, 4, 6)
        plt.plot(freqs_red, fft_red, color='orange')
        plt.title("After Noise Reduction (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        plt.subplot(2, 4, 7)
        plt.plot(freqs_fil, fft_fil, color='purple')
        plt.title("After Filtering (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        plt.subplot(2, 4, 8)
        plt.plot(freqs_fin, fft_fin, color='green')
        plt.title("After Normalization (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        plt.tight_layout()
        os.makedirs("test/plots", exist_ok=True)
        plt.savefig(os.path.join("test/plots", "data_full_pipeline.png"))
        plt.close()


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
