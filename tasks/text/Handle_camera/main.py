import cv2


class handle_camera():
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def stream(self):     
        # Check if the camera opened successfully
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv2.imshow('Live Stream', frame)

            cv2.waitKey(1)

            if cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) <1:
                break

        # When everything is done, release the capture and close any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()