from main import face_landmark


if __name__ == "__main__":
    
    image_path = "portrait.jpg"
    
    landmark = face_landmark()
    img = landmark.detect_landmark(image_path)
    img.save("Face_landmarked.jpg")

