import streamlit as st
import pandas as pd
from autocomplete import text_autocomplete
import time

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
# masker = profanity_masker()

# Streamlit app
st.title("🪄 AutoComplete")

# Description table as a single dictionary
description_table = {
    "Component": "Autocomplete",
    "Category":"N/A",
}

description_table2 = {
    "Model": "bert-base-uncased",
    "property":"N/A",
}

message = """
component text_autocomplete{
    service get_prediction{
        /**
        * Generates autocomplete suggestions for the input text.
        *
        * @param input_text The text for which suggestions need to generated.
        * @param top_k the maximum number of suggestions with the highest probability
        */

        [in] string input_text;
        [in] int top_k;
        [out] dict[pred_output] res;
        [out] int error_code;

    }
    struct pred_output{
        string Predictions;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

st.table(description_table2)

autocomplete = text_autocomplete()   
# Performance section
performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=100, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('sentences.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            res = autocomplete.get_prediction(sentence, 3)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            res = autocomplete.get_prediction(sentence, 3)

            
            
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

autocomplete = text_autocomplete()        

# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    user_input = st.text_input("Enter your sentence here:")
    if st.button("🪄 Predict"):
        if user_input:
            answer = []
            res = autocomplete.get_prediction(user_input, 3)
            # st.write(res["Predictions"].split("\n"))
            for i in res["Predictions"].split("\n"):
               answer.append(i)
            st.write(user_input + " " + answer[0])
            st.write(user_input + " " + answer[1])

            # new_input = st.text_input("input word: ")
        else:
            st.write("Please enter a sentence.")
