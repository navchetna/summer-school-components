import streamlit as st
import pandas as pd
import time
from pyannote.audio import Pipeline

   

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker

# Streamlit app
st.title("🗣️ Speech Segmentation")

# Description table as a single dictionary
description_table = {
    "Component": "Speech Segmentation",
    "Category":"N/A",
}

description_table2 = {
    "Model": "Pyannote.audio",
    "property":"N/A",
}

message = """
component speech_segmentation{
    service pipeline{
        [in] mp3_audio;
        [out] string text;
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
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=1, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=5, step=1)
    if st.button("Start Runs"):
        pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="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")
        default_audio = "karpathy.mp3"
        for i in range(warmup_criteria):
             
             diarization = pipeline(default_audio)
        # Load the CSV file
        # Extract the required number of sentences for warmup
        
        # Perform masking during the warmup phase without displaying anything
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            
             
            diarization = pipeline(default_audio)
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            
            # Start the timer for this run
            
            
            
            # Calculate performance metrics for this run
            
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / runs_criteria
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.2f} seconds",
    "Average Time Per Sentence": f"{average_time_per_sentence:.2f} seconds",
}

        st.table(description_table1)

        

# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    default_audio = "karpathy.mp3"
    user_input = default_audio
    st.write(default_audio)
    if st.button("🗣️ Click to Diarize"):
        if user_input:
            pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.0",
  use_auth_token="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")
            diarization = pipeline(user_input)
            st.write("Segmentation")
            st.write(diarization)
        else:
            st.write("Please enter a sentence.")
