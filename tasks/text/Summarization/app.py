import streamlit as st
import pandas as pd
from main import Summarizer
import time


# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
summarizer = Summarizer()

# Streamlit app
st.title("Summarizer")

# Description table as a single dictionary
description_table = {
    "Component": "Detect Language",
    "Category":"N/A",
}

description_table2 = {
    "Model": "facebook/bart-large-cnn",
    "property":"N/A",
    
}

message = """
component Summarizer{
    service summary_generator{
        [in] string input;
        [in] int max_summary_length;
        [out] string summary;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)


st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

st.write("Libraries Used")
st.table(description_table2)

performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=10, step=1)
    if st.button("Start Runs", key = "performance_button"):
        # Load the CSV file
        sentences_df = pd.read_csv('summaries.csv') 
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['Text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            summarizer.summary_generator(sentence)
            # pass
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['Text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            summarizer.summary_generator(sentence)
            
            
            # Calculate performance metrics for this run
            
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / runs_criteria
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.4f} seconds",
    "Average Time Per Sentence": f"{average_time_per_sentence:.4f} seconds",
}

        st.table(description_table1)


# Performance section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    
    user_input = st.text_input("Enter the sentence here")
    if st.button("Start Runs"):
        summary = summarizer.summary_generator(user_input)
        
        st.write(summary)
        
    
        
        
