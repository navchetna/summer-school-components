from main import denoise_audio

if __name__ == "__main__":
    
    try:
        denoise = denoise_audio()
        file_path = "noisy_audio.wav"
        denoise.remove_noise(file_path=file_path)
    
    except Exception as e:
        print("Error : ", e)