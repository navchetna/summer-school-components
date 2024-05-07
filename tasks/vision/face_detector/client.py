from face_detector_component import Face_Detector
import sys



if __name__ == "__main__": 
    MODEL_PATH = sys.argv[1]
    #MODEL_NAME = sys.argv[2]
    VEDIO_PATH = sys.argv[2]
    DEVICE = sys.argv[3]
    #LABEL_FILE = sys.argv[5]


    tsp_obj = Face_Detector()
    res = tsp_obj.face_detect(MODEL_PATH,VEDIO_PATH,DEVICE)
    print(res)

# Steps to run
#1.conda actiavte openvino
#2.python3 client.py ./models/face-detection-retail-0004.xml face-demographics-walking-and-pause.mp4 CPU  
