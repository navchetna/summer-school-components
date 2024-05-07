from main import detect_language



if __name__ == "__main__":
    
    text = ""  # Text to detect the language of

    '''
    Sample text 
    1. Otec matka syn. - Finnish 
    2. Buenas noches - Spanish
    3. Bonjour tout le monde - French
    4. Ein, zwei, drei, vier - German
    5. ありがと ございます - Japanese
    6. ನೀವು ಹೇಗಿದ್ದೀರಿ - Kannada
    7. के लिए एक कंप्यूटर विज्ञान पोर्टल है - Hindi
    '''
    language_detecter = detect_language()
    language = language_detecter.language_identifier(text)
    print("Detected Language: ", language)
    
    