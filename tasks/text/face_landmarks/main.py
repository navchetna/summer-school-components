import cv2
import dlib
from PIL import Image, ImageDraw
import numpy as np

class face_landmark:
    def __init__(self) -> None:
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.dlib_facelandmark = dlib.shape_predictor("my_model/shape_predictor_68_face_landmarks.dat")

    def detect_landmark(self, img_path):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        # Convert the image to grayscale
        gray = img.convert('L')

        # Detect faces
        faces = self.hog_face_detector(np.array(gray))
        for face in faces:
            # Detect landmarks
            face_landmarks = self.dlib_facelandmark(np.array(gray), face)
            # Draw landmarks
            for n in range(0, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(112, 235, 52))
                
        
        return img
