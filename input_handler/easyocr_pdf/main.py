import easyocr
import time

class OCRHandler:
    def __init__(self, language='en'):
        self.reader = easyocr.Reader([language]) 

    def read_text_from_image(self, image_path, detail=0):
        start_time = time.perf_counter()
        result = self.reader.readtext(image_path, detail=detail)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Result: ", result)
        print("Total Time: ", "%.2f" % total_time, " seconds")
        return result
