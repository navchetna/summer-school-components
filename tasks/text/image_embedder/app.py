import streamlit as st
import pandas as pd
from main import image_embedder
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
# detector = detect_language()

# Streamlit app
st.title("🖼️ Image Embedder ")

# Description table as a single dictionary
description_table = {
    "Component": "Image Embedder",
    "Category":"N/A",
}

description_table2 = {
    "Model": "RESNET50",
    "property":"N/A",
    
}

message = """
component image_embedder{
    service embed_image{
        [in] string image_path;
        [out] List embedding_vector;
        [out] int error code;
    }
    
    service calculate_similarity{
        [in] List first_image_embedding;
        [in] List second_image_embedding;
        [out] int similarity_score;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Model Used")
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

embedder = image_embedder()





performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    
    uploaded_file1 = st.file_uploader("choose an image", type=["png", "jpg"], key = "Performance_img1")
    if(st.button("Start", key = "Performance")):
        
        start_time = time.time()
        embedding1 = embedder.embed_image(uploaded_file1)
        
        
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
    uploaded_file2 = st.file_uploader("choose another image to compare to", type=["png", "jpg"])
    if(st.button("Start")):
        
        if uploaded_file1 and uploaded_file2:
            embedding1 = embedder.embed_image(uploaded_file1)
            # st.write("First Image")
            # st.image(uploaded_file1)
            
            embedding2 = embedder.embed_image(uploaded_file2)
            # st.write("Second Image")
            # st.image(uploaded_file2)

            similarity = embedder.calculate_similarity(embedding1, embedding2)
            
            result = 0 if similarity  <= 0 else similarity
            
            st.write(f"Similarity Score between the two images : {result :.3f}",)

            col1, col2 = st.columns(2)
            with col1:
                st.write("First Image")
                st.image(uploaded_file1)
            with col2:
                st.write("Second Image")
                st.image(uploaded_file2)
        else:
            st.write("Add the Images")