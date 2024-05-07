import streamlit as st
import pandas as pd
from main import Speech2Text
import time

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
speech = Speech2Text()
# Streamlit app
st.title("🎙️ Speech2Text")

# Description table as a single dictionary
description_table = {
    "Component": "Speech2Text",
    "Category":"N/A",
}

description_table2 = {
    "Model": "speech_recognition",
    "property":"N/A",
}

message = """
component speech2text{
    service convert_audio_to_text{
        [in] string audio_file_path;
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
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=100, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        default_audio = "speech1.wav"
        for i in range(warmup_criteria):
            speech.convert_audio_to_text(default_audio)


        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            speech.convert_audio_to_text(default_audio)

            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            
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

        

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Functionality expander
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    default_audio = "speech1.wav"
    default_sentence = "hey hope you are doing fine I wanted information regarding the application process"
    user_input = st.text_input("Text present in the default audio:", default_sentence, disabled=True)
    if st.button("🎙️Convert"):
        if user_input:
            audio = speech.convert_audio_to_text(default_audio)
            st.write("Converted Text:")
            st.write(audio)
        else:
            st.write("Please enter a sentence.")

