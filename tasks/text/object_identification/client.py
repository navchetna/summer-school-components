from main import object_detection


if __name__ == "__main__":
    detecter = object_detection()
    
    detecter.detect_objects(img_path="street_crossing.jpg", output_img_path="street_crossing_detected.jpg")