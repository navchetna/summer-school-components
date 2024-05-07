from main import Text2Speech
import soundfile as sf

if __name__ == '__main__':
    model = Text2Speech()
    speech = model.generate_audio("Hey, Hope you are doing fine. I wanted information regarding the application process.")
    sf.write("speech1.wav", speech["audio"], samplerate=speech["sampling_rate"])
    print('Sound file generated successfully!')