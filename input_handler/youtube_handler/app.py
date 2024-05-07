import streamlit as st
import pandas as pd
from main import YouTubeTranscriptExtractor
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
generator = YouTubeTranscriptExtractor()

# Streamlit app
st.title("▶️ Youtube Handler")

# Description table as a single dictionary
description_table = {
    "Component": "Youtube Handler",
    "Category":"N/A",
}

description_table2 = {
    "Library": "YouTubeTranscriptApi",
    "property":"N/A",
}

message = """
component YouTubeTranscriptExtractor{
    service fetch_transcript{
        /**
        * Extracts the transcript from the youtube video given the youtube_id.
        *
        * @param youtube_id The youtube_id of the video.
        */

        [in] string youtube_id;
        [out] List[Dict] transcript;
        [out] int error_code;
    };    
    struct Dict{
        string transcribed_text;
        int start;
        int duration;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)

# Print the message with the same indentation and format

st.table(description_table2)
st.write("Interface Definition Language (IDL)")
st.code(message, language='plaintext')

# Performance section
performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=1, value=5, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=50, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('reviews.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            generator.extract_text(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            generator.extract_text(sentence)
            
            
            # Calculate performance metrics for this run
            
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / runs_criteria
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.3f} seconds",
    "Average Time Per Sentence": f"{average_time_per_sentence:.3f} seconds",
}

        st.table(description_table1)

prompts = ["6M5VXKLf4D4",
            "ad79nYk2keg",
            "z5nc9MDbvkw",
            "Me3ea4nUt0U",
            "sQuFl0PSoXo",
            "6M5VXKLf4D4",
            "ad79nYk2keg",
            "z5nc9MDbvkw"
        ]

# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    if st.button("Select Random Youtube ID"):
        # Randomly select a sample text from the description_table3
        random_prompt = random.choice(prompts)
        
        # Fill the session state variables with the selected sample text
        st.session_state.user_input = random_prompt
    user_input = st.text_input("Enter your prompt", value = st.session_state.get("user_input", ""))
    if st.button("Generate Transcript"):
        if user_input:
            Extracted_Transcript = generator.extract_text(user_input)
            st.write("Extracted Transcript")
            st.write(Extracted_Transcript)
        else:
            st.warning("Please enter a Youtube ID.")