from landmark_detection import landmark_detector
import sys



if __name__ == "__main__": 
    MODEL_PATH_FACE_DETECT = sys.argv[1]
    MODEL_PATH_LANDMARK=sys.argv[2]
    #MODEL_NAME = sys.argv[2]
    VEDIO_PATH = sys.argv[3]
    DEVICE_FACE_DETECT = sys.argv[4]
    DEVICE_LANDMARK = sys.argv[4]
    #LABEL_FILE = sys.argv[5]


    tsp_obj = landmark_detector()
    res = tsp_obj.landmark_detect(MODEL_PATH_FACE_DETECT,MODEL_PATH_LANDMARK,VEDIO_PATH,DEVICE_FACE_DETECT,DEVICE_LANDMARK)
    print(res)

#steps to run
#1.conda activate openvino
#2.python client.py ./models/face-detection-retail-0004.xml ./models/landmarks-regression-retail-0009.xml face-demographics-walking-and-pause.mp4 CPU CPU 
