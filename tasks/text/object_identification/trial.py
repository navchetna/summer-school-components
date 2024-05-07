import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import numpy as np


class object_detection():
    def __init__(self) -> None:
        None

    def detect_objects(self, img_path):

        

        img = Image.open(img_path)
        labels = []

        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Bounding box.
        # the cvlib library has learned some basic objects using object learning
        # usually it takes around 800 images for it to learn what a phone is.
        bbox, label, conf = cv.detect_common_objects(image)

        output_image = draw_bbox(image, bbox, label, conf)       

        for item in label:
            if item in labels:
                pass
            else:
                labels.append(item)
        

                
        return output_image, labels
