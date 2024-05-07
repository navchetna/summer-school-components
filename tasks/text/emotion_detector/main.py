import cv2
from deepface import DeepFace
from PIL import Image
import numpy as np


class emotion_detector():
    def __init__(self) -> None:
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_emotion(self, img_path):

        pil_image = Image.open(img_path)

        # Convert PIL image to OpenCV format
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]
         
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion'] 
            
            return emotion       