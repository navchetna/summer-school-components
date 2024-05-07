import streamlit as st
import io
from main import ImageTextExtractor
import time
from PIL import Image
import numpy as np
import cv2

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
extract = ImageTextExtractor()

# Streamlit app
st.title("Tesseract")

# Description table as a single dictionary
description_table = {
    "Component": "Tesseract",
    "Category":"N/A",
}

description_table2 = {
    "Library": "pytesseract",
    "property":"N/A",
    
}

message = """
component ImageTextExtractor{
    service extract_text{
        /**
        * Extracts text from the image
        * 
        * @param image_path string containing the path to the image.
        * @param text the content of the input image.
        */

        [in] string image_path;
        [out] string text;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Model Used")
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')


performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    
    images = ["pan_card1.jpg", "pan_card2.jpg", "pan_card3.jpg"]
    warmup = st.number_input("number of Test runs" ,min_value=5)
    test_runs = st.number_input("number of Test runs" ,min_value=50)
    
    if st.button("Start Runs"):
        
        # Prepare to collect metrics for the runs criteria loop
        
        for i in range(warmup):
            img_idx = i % len(images)
            extract.extract_text(cv2.imread(images[img_idx])) 
                
        # Prepare to collect metrics for the runs criteria loop
        
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(images)
            extract.extract_text(cv2.imread(images[img_idx]))        

            
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
    
    uploaded_file1 = st.file_uploader("choose an image", type=["png", "jpg"])
    
    if(st.button("Start")):
        
        if uploaded_file1:
            
            image = Image.open(io.BytesIO(uploaded_file1.read()))
            image_np = np.array(image)

            res = extract.extract_text(image_np)   
            # st.write("First Image")
            # st.image(uploaded_file1)
            
            st.write(res)
            
        else:
            st.write("Add the Images")