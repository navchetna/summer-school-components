from action_recognition_demo import action_recognition
import sys

if __name__ == "__main__": 
    MODEL_EN = sys.argv[1]
    MODEL_DE = sys.argv[2]
    AT=sys.argv[3]
    VEDIO_PATH = sys.argv[4]
    LABEL_FILE=sys.argv[5]
    DEVICE = sys.argv[6]

    
    tsp_obj = action_recognition()
    res = tsp_obj.recognize(MODEL_EN,MODEL_DE,AT,VEDIO_PATH,LABEL_FILE,DEVICE)
    print(res)

#steps to run 
#1.conda activate openvino
#2.python client.py ./models/driver-action-recognition-adas-0002-encoder.xml ./models/driver-action-recognition-adas-0002-decoder.xml en-de driver-action-recognition.mp4 /home/develop/open_model_zoo/demos/action_recognition_demo/python/driver_actions.txt CPU


