from imageai.Detection import ObjectDetection


class object_detection():
    def __init__(self) -> None:
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("yolov3.pt")
        self.detector.loadModel()

    def detect_objects(self, img_path, output_img_path):

        detections = self.detector.detectObjectsFromImage(input_image=img_path, output_image_path=output_img_path)
        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"])
