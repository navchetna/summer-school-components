import streamlit as st
import pandas as pd
from main import Text2Speech
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
generator = Text2Speech()

# Streamlit app
st.title("Text2Speech")

# Description table as a single dictionary
description_table = {
    "Component": "Text to Speech",
    "Category":"N/A",
}

description_table2 = {
    "Model": "microsoft/speecht5_tts",
    "property":"N/A",
}

message = """
component Text2speech{
    service generate_audio{
        [in] string prompt;
        [out] audio_file generated_audio;
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
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=4, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('reviews.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            generator.generate_audio(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            generator.generate_audio(sentence)
            
            
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

prompts = ["Life is an invaluable gift that we are bestowed with. a journey filled with unique experiences and opportunities for growth.", 
            "The essence of life's preciousness lies in its unpredictability and the myriad of emotions it evokes. Every moment whether joyful or challenging contributes to our personal evolution and understanding of the world around us.",
            "Life's beauty is not just in its highs but also in its lows teaching us resilience empathy and the significance of gratitude. It is in the intricate tapestry of life's ups and downs that we find meaning purpose and the chance to make a positive impact on others.",
            "Embracing life with all its complexities cherishing each day as a gift and valuing the relationships and experiences that shape us underscores the profound preciousness of life.",
            "An ecosystem is a complex network where living organisms interact with each other and their physical environment forming a delicate balance of life. ",
            "It comprises both biotic components such as plants animals and microorganisms and abiotic factors like air water soil and sunlight. These components work together through nutrient cycles and energy flows regulating essential ecological processes supporting life systems and maintaining stability.",
            "Ecosystems play a crucial role in cycling nutrients between biotic and abiotic elements balancing trophic levels and synthesizing organic components through energy exchange. They provide a variety of goods and services vital for human survival such as clean air water food and habitat highlighting the intricate interdependence of all organisms within an ecosystem."
            ]

# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    if st.button("Select Random prompt"):
        # Randomly select a sample text from the description_table3
        random_prompt = random.choice(prompts)
        
        # Fill the session state variables with the selected sample text
        st.session_state.user_input = random_prompt
    user_input = st.text_input("Enter your prompt", value = st.session_state.get("user_input", ""))
    if st.button("Generate Audio"):
        if user_input:
            generated_audio = generator.generate_audio(user_input)
            st.write("Generated Audio:")
            st.audio(generated_audio['audio'], sample_rate=generated_audio['sampling_rate'])
        else:
            st.write("Please enter a sentence.")
