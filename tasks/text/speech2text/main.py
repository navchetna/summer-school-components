import speech_recognition as sr

class Speech2Text():
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def convert_audio_to_text(self, audio_file):
        # Load audio file
        with sr.AudioFile(audio_file) as source:
            # Listen for the audio file and extract it into audio data
            audio_data = self.recognizer.record(source)

            try:
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
                # Return the recognized text
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None  # Return None or an appropriate value to indicate failure
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                return None  # Return None or an appropriate value to indicate failure


