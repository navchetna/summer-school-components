from main import speech2text

if __name__ == "__main__":
    convert = speech2text()
    audio_file = "speech1.wav"  # Replace with the path to your audio file
    convert.convert_audio_to_text(audio_file)