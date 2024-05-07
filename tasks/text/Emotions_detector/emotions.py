import cv2
from deepface import DeepFace
import time

def detect_emotion(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        print("Error loading image")
        return
    
    # Detect the emotion using DeepFace with enforce_detection set to False
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    
    # The result is a list containing a dictionary, so we access the first element and then the 'dominant_emotion' key
    emotion = result[0]['dominant_emotion']
    
    # Print the detected emotion
    print(f"Detected emotion: {emotion}")

# Example usage
if __name__ == "__main__":
    start = time.time()
    image_path = 'neutral.jpg' # Replace with your image path
    detect_emotion(image_path)
    print(time.time() - start)