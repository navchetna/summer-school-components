import streamlit as st
import pandas as pd
from main import handle_camera
import cv2

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
# detector = detect_language()

# Streamlit app
st.title("Handle Camera")

# Description table as a single dictionary
description_table = {
    "Component": "Handle Camera",
    "Category":"N/A",
}

description_table2 = {
    "Library 1": "open-cv",
    "Library 2": "PIL",
    "property":"N/A",
    
}

message = """
component handle_camera{
    service stream{
        [in] Live camera input;
        [out] Live Streaming
        [out] int error_code;
    }
}
"""



def stream():     
        # Check if the camera opened successfully
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error opening video stream or file")
                return

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

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
            cap.release()
            cv2.destroyAllWindows()



# Display the table with all details in the first row
st.table(description_table)

st.write("Model Used")
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

handler = handle_camera()








# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    
    if(st.button("Start")):
        stream()
        