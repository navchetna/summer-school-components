import streamlit as st
import io
from main import OCRHandler
import time
from PIL import Image
import numpy as np

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
ocr = OCRHandler()

# Streamlit app
st.title("EasyOCR")

# Description table as a single dictionary
description_table = {
    "Component": "EasyOCR",
    "Category":"N/A",
}

description_table2 = {
    "Library": "EasyOCR",
    "property":"N/A",
    
}

message = """
component OCRHandler{
    service read_text_from_image{
        /**
        * Reads text from an image file.
        * 
        * @param image_path The path to the image file to be processed.
        * @param detail The level of detail for text extraction (0 for basic, 1 for detailed).
        * @param result The extracted text from the image.
        */

        [in] string  image_path;
        [in] int detail;
        [out] string result;  
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
            ocr.read_text_from_image(images[img_idx]) 
                
        # Prepare to collect metrics for the runs criteria loop
        
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(images)
            ocr.read_text_from_image(images[img_idx])       

            
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

            res = ocr.read_text_from_image(image_np)     
            # st.write("First Image")
            # st.image(uploaded_file1)
            
            st.write(res)
            
        else:
            st.write("Add the Images")