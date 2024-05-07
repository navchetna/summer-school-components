from main import EntityRecognition

if __name__ == "__main__":
    
    try:
        recognizer = EntityRecognition()
        
        text = "Rob is a very active employee at Intel"
        
        recognizer.recognize_entities(input_text=text)
    except Exception as e:
        print("Error : ", e)
    