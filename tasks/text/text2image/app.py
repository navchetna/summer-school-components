import streamlit as st
import pandas as pd
from main import Text2Image
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
generator = Text2Image()

# Streamlit app
st.title("Text 2 Image")

# Description table as a single dictionary
description_table = {
    "Component": "Text to Image",
    "Category":"N/A",
}

description_table2 = {
    "Model": "runwayml/stable-diffusion-v1-5",
    "property":"N/A",
}

message = """
component Text2Image{
    service generate{
        [in] string prompt;
        [out] int error_code;
    };
};
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
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=2, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=5, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('reviews.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            generator.generate(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            generator.generate(sentence)
            
            
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


prompts = ["Cars racing on Jupyter.",
            "A big flower bed with a lot of roses",
            "A scenic windmill",
            "Water trickling down the mountains",
            "Grandpa in a space suit",
            "Tall tower in the middle of nowhere",
            "Mountain with chocolate water fall",
            "Alligators in desert"
            ]

# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    
    if st.button("Select Random prompt"):
        # Randomly select a sample text from the description_table3
        random_prompt = random.choice(prompts)
        
        # Fill the session state variables with the selected sample text
        st.session_state.user_input = random_prompt
        
    user_input = st.text_input("Enter Prompt", value = st.session_state.get("user_input", ""))
    if st.button("Generate Image"):
        if user_input:
            generated_image = generator.generate(user_input)
            st.write("Generated_image")
            st.image(generated_image)
        else:
            st.write("Please enter a prompt.")
