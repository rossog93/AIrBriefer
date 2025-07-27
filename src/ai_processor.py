# src/ai_transcriber.py
import whisper
import logging
import noisereduce as nr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
from src.utils.logger import logger

import os

class AITranscriber:
    def __init__(self, model_name, language="es"):
        self.language = language
        logging.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def reduce_noise(self, input_path, plt=True, low_pass_freq=500, high_pass_freq=2500):
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
        logger.info(f"Applying band-pass filter ({low_pass_freq}–{high_pass_freq} Hz).")
        filtered_audio = clean_audio.high_pass_filter(low_pass_freq).low_pass_filter(high_pass_freq)
        filtered_samples = np.array(filtered_audio.get_array_of_samples())

        # Step 3: Normalization
        logger.info("Normalizing audio amplitude.")
        normalized_audio = filtered_audio.normalize()
        final_samples = np.array(normalized_audio.get_array_of_samples())

        logger.info("Noise reduction, filtering, and normalization completed. Exporting clean audio.")
        format = input_path.split(".")[-1]
        output_path = input_path.split(".")[0] + f"_clean.{format}"
        normalized_audio.export(output_path, format=f"{format}")
        logger.info("Clean audio exported successfully.")

        if plt:
            self.plot_signals(
                original=samples,
                reduced=reduced_noise,
                filtered=filtered_samples,
                final=final_samples,
                sample_rate=audio.frame_rate
            )

    def plot_signals(self, original, reduced, filtered, final, sample_rate):
        import matplotlib.pyplot as plt

        def compute_fft(signal):
            freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
            fft_vals = np.abs(np.fft.rfft(signal))
            return freqs, fft_vals

        def compute_psd(signal):
            freqs, psd_vals = welch(signal, fs=sample_rate, nperseg=1024)
            return freqs, psd_vals

        signals = [
            ("Original", original, 'blue'),
            ("Noise Reduced", reduced, 'orange'),
            ("Filtered", filtered, 'purple'),
            ("Normalized", final, 'green'),
        ]

        fig, axs = plt.subplots(4, 4, figsize=(24, 14))
        axs = axs.flatten()

        for i, (label, signal, color) in enumerate(signals):
            time = np.arange(len(signal)) / sample_rate

            # Time Domain
            axs[i].plot(time, signal, color=color)
            axs[i].set_title(f"{label} - Time")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Amplitude")
            axs[i].grid(True)

            # FFT
            freqs_fft, fft_vals = compute_fft(signal)
            axs[i + 4].plot(freqs_fft, fft_vals, color=color)
            axs[i + 4].set_title(f"{label} - FFT")
            axs[i + 4].set_xlabel("Frequency (Hz)")
            axs[i + 4].set_ylabel("Magnitude")
            axs[i + 4].grid(True)

            # PSD
            freqs_psd, psd_vals = compute_psd(signal)
            axs[i + 8].semilogy(freqs_psd, psd_vals, color=color)
            axs[i + 8].set_title(f"{label} - PSD")
            axs[i + 8].set_xlabel("Frequency (Hz)")
            axs[i + 8].set_ylabel("PSD (V²/Hz)")
            axs[i + 8].grid(True)

            # Histogram
            axs[i + 12].hist(signal, bins=100, color=color, alpha=0.7)
            axs[i + 12].set_title(f"{label} - Amplitude Histogram")
            axs[i + 12].set_xlabel("Amplitude")
            axs[i + 12].set_ylabel("Count")

        plt.tight_layout()

        plot_file = os.path.join("test", "data_full_pipeline.png")
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Saved plot to {plot_file}")


    def transcribe(self, audio_file, reduce_noise=False):
        try:
            logging.info(f"Transcribing audio: {audio_file}")

            if reduce_noise:
                logging.info("Noise reduction enabled, processing clean audio.")
                self.reduce_noise(audio_file)

            result = self.model.transcribe(audio_file,language="es")
            transcription = result["text"]
            logging.info(f"Transcription completed: {transcription}")
            return transcription

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""
