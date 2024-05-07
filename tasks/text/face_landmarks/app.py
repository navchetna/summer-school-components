import streamlit as st
import pandas as pd
from main import face_landmark
import time
import random
import cv2

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
detector = face_landmark()

# Streamlit app
st.title("Face Landmark")

# Description table as a single dictionary
description_table = {
    "Component": "Face Landmark detection",
    "Category":"N/A",
}

description_table2 = {
    "Model": "shape_predictor_68_face_landmarks.dat",
    "property":"N/A",
}

message = """
component face_landmark{
    service detect_landmark{
        [in] string image_path;
        [out] image detected_landmark;
        [out] int error_code;
    }
}
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
    images = ["portrait1.jpg", "portrait2.jpg", "portrait3.jpg"]
    
    warmup = st.number_input("number of Test runs" ,min_value=10)
    test_runs = st.number_input("number of Test runs" ,min_value=100)
    if st.button("Start Runs"):
        
        # Prepare to collect metrics for the runs criteria loop
        
        for i in range(warmup):
            img_idx = i % len(images)
        
            detector.detect_landmark(img_path=images[img_idx])       
        
        # Prepare to collect metrics for the runs criteria loop
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(images)
        
            detector.detect_landmark(img_path=images[img_idx])       

            
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
    if st.button("Generate landmark"):
        if uploaded_file2:
            img = detector.detect_landmark(uploaded_file2)
            
            st.write("Generated Image")
            st.image(img)
            
        else:
            st.warning("Please input an image.")
