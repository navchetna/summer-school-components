import streamlit as st
import pandas as pd
from main import synonym_generator
import time

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
synonyms = synonym_generator()

# Streamlit app
st.title("💡 Synonyms Generator")

# Description table as a single dictionary
description_table = {
    "Component": "Synonyms Generator",
    "Category":"N/A",
}

description_table2 = {
    "Model": "from nltk import wordnet",
    "property":"N/A",
}

message = """
component synonym_generator{
    service find_synonyms{
        [in] string word;
        [out] List synonyms;
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
        sentences_df = pd.read_csv('reviews.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['words'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for word in warmup_sentences:
                synonyms1 = synonyms.find_synonyms(word)            

        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['words'].sample(1).iloc[0]
            
            # Start the timer for this run
            synonyms1 = synonyms.find_synonyms(word)            

            
            
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
    word = st.text_input("Enter your word here:")
    if st.button("💡 Generate"):
        if word:
            synonyms = synonyms.find_synonyms(word)            
            st.write("Synonymns:")
            st.write(synonyms)
        else:
            st.write("Please enter a word.")
