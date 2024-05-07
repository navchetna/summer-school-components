import streamlit as st
import io
from main import get_paragraphs
import time
from PIL import Image
import numpy as np
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker


# Streamlit app
st.title("HTML Data Extractor")

# Description table as a single dictionary
description_table = {
    "Component": "HTML Data Extractor",
    "Category":"N/A",
}

description_table2 = {
    "Library 1": "HTMLParser",
    "Library 2": "urllib", 
    "property":"N/A",
    
}

message = """
component html_handler{
    service get_paragraphs{
        /**
            * Extracts paragraphs from webpages.
            * 
            * @param url_list   List of URLs from which data (paragraphs) needs to extracted.
            * @param all_paragraphs  List of strings containing the extracted data from URLs.
            */

            [in] List[string] url_list;
            [out] List[string] all_paragraphs;
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
    
    url_list = [
        "https://en.wiktionary.org/wiki/Wiktionary:Main_Page",
        "https://www.nasa.gov/", 
        "https://www.python.org/",  
        "https://www.reddit.com/", 
        "https://www.w3schools.com/"
    ]
    warmup = st.number_input("number of Test runs" ,min_value=5)
    test_runs = st.number_input("number of Test runs" ,min_value=50)
    
    if st.button("Start Runs"):
        
        # Prepare to collect metrics for the runs criteria loop
        
        for i in range(warmup):
            img_idx = i % len(url_list)
            
            get_paragraphs(url_list[img_idx]) 
                
        # Prepare to collect metrics for the runs criteria loop
        
        start_time = time.time()
        for i in range(test_runs):
            img_idx = i % len(url_list)
            get_paragraphs(url_list[img_idx])       

            
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
    
    if(st.button("Select Random URL")):
        random_url = random.choice(url_list)
        st.session_state.url = random_url
        
    url_input = st.text_input("Enter a URL", value=st.session_state.get("url"))
    if(st.button("Start")):
        
        if url_input:
            res = get_paragraphs([url_input])
            
            st.write(res)
        
        else:
            
            st.write("Add an URL")