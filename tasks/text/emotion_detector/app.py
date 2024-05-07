import streamlit as st
import pandas as pd
from main import emotion_detector
import time
import random
import cv2

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
detector = emotion_detector()

# Streamlit app
st.title("😊 Emotion Detection")

# Description table as a single dictionary
description_table = {
    "Component": "Emotion Detection",
    "Category":"N/A",
}

description_table2 = {
    "Model": "haarcascade_frontalface_default.xml",
    "property":"N/A",
}

message = """
component emotion_detector{
    service detect_emotion{
        [in] string img_path;
        [out] string detected_emotion;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

st.table(description_table2)

# Performance section
performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    
    images = ["angry_man.jpg", "sad_man.jpg", "happy_man.jpg"]
    warmup = st.number_input("Enter warmup criteria:", min_value=1, value=5, step=1)
    test_runs = st.number_input("number of Test runs" ,min_value=100)
    if st.button("Start Runs"):
        for i in range(warmup):
            img_idx = i % len(images)
            detector.detect_emotion(img_path=images[img_idx])   
        
        # Prepare to collect metrics for the runs criteria loop
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(images)
            detector.detect_emotion(img_path=images[img_idx])       
        
        
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / test_runs
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.4f} seconds",
    "Average Time Per Image": f"{average_time_per_sentence:.4f} seconds",
}
        st.table(description_table1)



# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    uploaded_file2 = st.file_uploader("choose an image", type=["png", "jpg"])
    if st.button("Detect"):
        if uploaded_file2:
            emotion = detector.detect_emotion(uploaded_file2)
            
            st.write(f"Detected Emotion : {emotion}")
            st.image(uploaded_file2)
            
        else:
            st.warning("Please input an image.")
