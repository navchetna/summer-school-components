from main import emotion_detector
import time

if __name__ == "__main__":
    detector = emotion_detector()
    start = time.time()
    emotion = detector.detect_emotion("angry_man.jpg")
    print("Time taken : ",time.time() - start)
    print("Detected Emotion : ", emotion)