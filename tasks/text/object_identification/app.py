import streamlit as st
import pandas as pd
from trial import object_detection
import time
import random
import cv2 
import numpy as np  

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
detector = object_detection()

# Streamlit app
st.title("Object Identification")

# Description table as a single dictionary
description_table = {
    "Component": "Object Identification",
    "Category":"N/A",
}

description_table2 = {
    "Library": "cvlib",
    "property":"N/A",
}

message = """
component object_detection{
    service detect_objects{
        [in] string img_path;
        [in] string output_img_path;
        [out] int error code;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

st.table(description_table2)



performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    
    uploaded_file1 = st.file_uploader("choose an image", type=["png", "jpg"], key = "Performance_img1")
    if(st.button("Start", key = "Performance")):
        
        start_time = time.time()
        output_img = detector.detect_objects(uploaded_file1)
    
        end_time = time.time() - start_time
        
        performance_table = {
                "Total Time taken " : f"{end_time:.3f} seconds", 
                "Average Time for one image" : f"{(end_time / 2):.3f} seconds"
        }
        
        st.table(performance_table)
        
        





# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    uploaded_file1 = st.file_uploader("choose an image", type=["png", "jpg"])
    
    
    if(st.button("Start")):
        
        if uploaded_file1:
            detected_img, labels = detector.detect_objects(uploaded_file1)
            st.code(labels)

            st.write("Detected Image")
            st.image(detected_img)
            

        else:
            st.write("Add the Images")