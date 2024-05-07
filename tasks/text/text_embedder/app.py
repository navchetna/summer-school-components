import streamlit as st
import pandas as pd
from main import embedder
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the autocorrect
text_embedder = embedder()

# Streamlit app
st.title("🔤 Text Embedder")

# Description table as a single dictionary
description_table = {
    "Component": "Text Embedder",
    "Category":"N/A",
}

description_table2 = {
    "Model": "all-MiniLM-l6-v2",
    "property":"N/A",
}

message = """
component embedder{
    service embed_text{
        /**
        * Generates the embedding for the given text. Text could be any string.
        *
        * @param text The text to create an embedding_vector for.
        * @param embedding_vector the embedding_vector for the input text.
        */

        [in] string text;
        [out] List[int] embedding_vector; 
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
    if st.button("Start"):
        # Load the CSV file
        sentences_df = pd.read_csv('sentences.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['Text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            text_embedder.embed_text(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['Text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            text_embedder.embed_text(sentence)
            
            
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

    
sentences = {
1: "He is moving here",
2: "They are going to the beach",
3: "He is reading a book",
4: "The cat is sleeping on the chair.",
5: "We are going to the park.",
6: "The dog is barking at her.",
7: "I am going to the party.",
8: "The sun is shining brightly.",
9: "They are watching a movie.",
10: "She goes to school by bus.",
11: "He is playing with them.",
12: "We are eating dinner now."
}

functionality = st.expander("Functionality", expanded=False)
df = pd.DataFrame.from_dict(sentences, orient='index', columns=["Sample Text"])
with functionality:
    
    st.table(df)
    
    if st.button("Select Random Sample Text"):
        # Randomly select a sample text from the description_table3
        random_key = random.choice(list(sentences.keys()))
        random_text = sentences[random_key]
        
        # Fill the user_input with the selected sample text
        st.session_state.user_input = random_text
    
    # Create a text input field for the user input
    user_input = st.text_input("Enter the sentence here", key="user_input")
 
    if st.button("Start", key = "Functionality"):
        
        if user_input:
            emdedding = text_embedder.embed_text(user_input)
            
            st.write("emdedding")
            
            st.code(emdedding)
        else:
            st.write("Please enter the text")