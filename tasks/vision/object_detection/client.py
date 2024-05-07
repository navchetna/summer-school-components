from object_detection import object_detection
import sys



if __name__ == "__main__": 
    MODEL_PATH = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    VEDIO_PATH = sys.argv[3]
    DEVICE = sys.argv[4]
    LABEL_FILE = sys.argv[5]


    tsp_obj = object_detection()
    res = tsp_obj.object_detector(MODEL_PATH,MODEL_NAME,VEDIO_PATH,DEVICE,LABEL_FILE)
    print(res)