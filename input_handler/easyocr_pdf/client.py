from main import OCRHandler

def main():
    ocr_handler = OCRHandler()
    image_path = 'MicrosoftTeams-image.png'
    ocr_handler.read_text_from_image(image_path)

if __name__ == "__main__":
    main()
