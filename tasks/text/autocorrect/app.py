import streamlit as st
import pandas as pd
from main import AutoCorrect
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the autocorrect
autocorrect = AutoCorrect()

# Streamlit app
st.title("AutoCorrect")

# Description table as a single dictionary
description_table = {
    "Component": "English AutoCorrect",
    "Category":"N/A",
}

description_table2 = {
    "Model": "deep-learning-analytics/GrammarCorrector",
    "property":"N/A",
}

message = """
component AutoCorrect{
    service check_grammar{
        [in] string input_text;
        [out] float score;
        [out] int error_code;
    };
    service correct_grammar{
        [in] string input_text;
        [in] int max_suggestions;
        [out] List[string] suggestions;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')




performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=50, step=1)
    max_suggestions = st.number_input("Maximum number of suggestions", min_value = 2)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('grammatical_error.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['Text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            autocorrect.correct_grammar(sentence, max_suggestions)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['Text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            autocorrect.correct_grammar(sentence, max_suggestions)
            
            
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

    
incorrect_sentences = {
1: "He are moving here",
2: "They is going to the beach",
3: "He am reading a book",
4: "The cat are sleeping on the chair.",
5: "We is going to the park.",
6: "The dog is barking at she.",
7: "I are going to the party.",
8: "The sun are shining brightly.",
9: "They am watching a movie.",
10: "She go to school by bus.",
11: "He is playing with they.",
12: "We am eating dinner now."
}

functionality = st.expander("Functionality", expanded=False)

with functionality:
    
    
    if st.button("Select Random Sample Text"):
        # Randomly select a sample text from the description_table3
        random_key = random.choice(list(incorrect_sentences.keys()))
        random_text = incorrect_sentences[random_key]
        
        # Fill the user_input with the selected sample text
        st.session_state.user_input = random_text
    
    # Create a text input field for the user input
    user_input = st.text_input("Enter the sentence here", key="user_input")
    max_suggestions = st.number_input("Number of suggestions to be shown: ",min_value= 2,  step=1, key="max_suggestions_functionality")
    
    if st.button("Start Run"):
        
        if user_input and max_suggestions:
            score = autocorrect.check_grammar(user_input)
            st.write("Score based on Grammatical correctness of the user's sentence input:", score)
            
            suggestions = autocorrect.correct_grammar(user_input, num_return_sequences=max_suggestions)
            
            st.write("Suggestions")
            
            for i in range(len(suggestions)):
                st.write(f"{i + 1}: " + suggestions[i])
        else:
            st.write("Please fill the above")
