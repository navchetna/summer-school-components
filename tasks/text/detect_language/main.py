from langdetect import detect
import pycountry


class detect_language:
    def language_identifier(self, text):  
        language_code = detect(text)  # Detect the language of the text
        language_name = pycountry.languages.get(alpha_2=language_code).name  # Get the full name of the language

        return language_name