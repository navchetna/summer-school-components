import streamlit as st
import io
from main import poseDetector
import time
import cv2
import numpy as np
from PIL import Image

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker


# Streamlit app
st.title("Pose Identifier")

# Description table as a single dictionary
description_table = {
    "Component": "Pose Identifier",
    "Category":"N/A",
}

description_table2 = {
    "Model": "graph_opt.pb",
    "property":"N/A",
}

message = """
component pose_estimation{
    serive poseDetector{
        [in] string image_path;
        [out] List[List] image;
        [out] int error_code;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

# Print the message with the same indentation and format

st.table(description_table2)
st.write("Interface Definition Language (IDL)")
st.code(message, language='plaintext')
performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    images = ["pose1.jpg", "pose2.jpg", "pose3.jpg"]
    
    warmup = st.number_input("number of Test runs" ,min_value=10)
    test_runs = st.number_input("number of Test runs" ,min_value=100)
    if st.button("Start Runs"):
        
        # Prepare to collect metrics for the runs criteria loop
        
        for i in range(warmup):
            img_idx = i % len(images)
        
            poseDetector(cv2.imread(images[img_idx]))       
        
        # Prepare to collect metrics for the runs criteria loop
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(images)
        
            poseDetector(cv2.imread(images[img_idx]))     

            
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
    if st.button("Identify Pose"):
        if uploaded_file2:
            image = Image.open(io.BytesIO(uploaded_file2.read()))
            image_np = np.array(image)
            img = poseDetector(image_np)
            
            st.write("Detected Pose")
            st.image(img)
            
        else:
            st.warning("Please input an image.")
