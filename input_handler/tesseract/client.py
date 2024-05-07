from main import ImageTextExtractor
import cv2

if __name__ == "__main__":
    extractor = ImageTextExtractor()
    image_path = "/home/develop/components/input_handler/tesseract/MicrosoftTeams-image.png"
    image = cv2.imread(image_path)
    extractor.extract_text(image)
