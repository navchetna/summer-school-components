import librosa
import soundfile
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from scipy import signal
import math
import matplotlib.pyplot as plt

class denoise_audio():
   
    def add_noise(self, file_path):
        # Adds gaussian white noise to an audio file
        
        signal, sr = librosa.load(file_path, sr = 16000)
        RMS=math.sqrt(np.mean(signal**2))
        STD_n= 0.04
        noise=np.random.normal(0, STD_n, signal.shape[0])
        signal_noise = signal+noise
        soundfile.write('noisy_audio.wav',signal_noise,16000)
        
        
    def remove_noise(self, file_path):
        
        rate, data = wavfile.read(file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        cutoff_freq = 4000  # Adjust the cutoff frequency as needed
        b, a = signal.butter(4, cutoff_freq / (rate / 2), 'low')
        filtered_audio = signal.filtfilt(b, a, reduced_noise)
        filtered_audio *= 10**(15/20)  # Applying a gain of 15 dB

        # Save the processed audio to a new file
        wavfile.write('output.wav', rate, np.int16(filtered_audio))
        self.compare(data, filtered_audio)
        print("Noise reduction completed successfully.")
        
    def compare(self, original_data, processed_data):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(original_data, color='b')
        plt.title('Original Audio Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 1, 2)
        plt.plot(processed_data, color='r')
        plt.title('Noise-Reduced Audio Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('audio_comparison.png')
        