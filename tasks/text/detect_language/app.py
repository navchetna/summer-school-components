import streamlit as st
import pandas as pd
from main import detect_language
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
detector = detect_language()

# Streamlit app
st.title(":earth_africa: Language Detector ")

# Description table as a single dictionary
description_table = {
    "Component": "Detect Language",
    "Category":"N/A",
}

description_table2 = {
    "Library 1": "langdetect",
    "Library 2": "pycountry",
    "property":"N/A",
    
}

message = """
component detect_language{
    service language_identifier{
        [in] string input_text;
        [out] string detected_language;
        [out] int error_code;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Libraries Used")
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

description_table3 = {
    "Kannada": "ನೀವು ಹೇಗಿದ್ದೀರಿ", 
    "Marathi": "नमस्कार, कसे आहात", 
    "Hindi": "आप कैसे हैं?", 
    "Gujarati": "હેલો, કેમ છો",
    "Tamil": 'நீ செய்கிறாய்',
    "Punjabi": "ਹੈਲੋ ਤੁਸੀ ਕਿਵੇਂ ਹੋ", 
    "Bengali": "হ্যালো, আপনি কেমন আছেন",
    "Japanese": "ありがと ございます", 
    "German": "Ein, zwei, drei, vier",
    "French": "Bonjour tout le monde",
    "Spanish": "Buenas noches",
    "Finnish": "Otec matka syn.",
    "Arabic" : "مرحبا، كيف حالك", 
    "Polish": "Witam, jak się masz",
    "Dutch": "Hallo hoe is het"
}
df = pd.DataFrame.from_dict(description_table3, orient='index', columns=["Sample Text"])

performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=100, step=1)
    if st.button("Start Runs", key = "performance_button"):
        # Load the CSV file
        sentences_df = pd.read_csv('reviews.csv') 
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            detector.language_identifier(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            detector.language_identifier(sentence)
            
            
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


# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    st.table(df)
    
    # Add a button to randomly select a sample text
    if st.button("Select Random Sample Text"):
        # Randomly select a sample text from the description_table3
        random_key = random.choice(list(description_table3.keys()))
        random_text = description_table3[random_key]
        
        # Fill the user_input with the selected sample text
        st.session_state.user_input = random_text
    
    # Create a text input field for the user input
    user_input = st.text_input("Enter the sentence here", key="user_input")
    
    if st.button("Start Runs"):
        # Load the CSV file
        
        detected_language = detector.language_identifier(user_input)
        st.write("Detected Language : ", detected_language)
