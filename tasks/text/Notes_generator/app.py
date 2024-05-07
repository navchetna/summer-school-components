import streamlit as st
import pandas as pd
from main import notes_generator
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
generator = notes_generator()

# Streamlit app
st.title("📝 Notes Generator")

# Description table as a single dictionary
description_table = {
    "Component": "Notes Generator",
    "Category":"N/A",
}

description_table2 = {
    "Library": "nltk",
    "property":"N/A",
}

message = """
component notes_generator{
    service generate_points{
        [in] String Summary;
        [out] string important_points;
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
        sentences_df = pd.read_csv('summary.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            generator.generate_points(sentence)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            generator.generate_points(sentence)
            
            
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

prompts = ["An ecosystem is a complex network where living organisms interact with each other and their physical environment forming a delicate balance of life. It comprises both biotic components such as plants animals and microorganisms and abiotic factors like air water soil and sunlight. These components work together through nutrient cycles and energy flows regulating essential ecological processes supporting life systems and maintaining stability. Ecosystems play a crucial role in cycling nutrients between biotic and abiotic elements balancing trophic levels and synthesizing organic components through energy exchange. They provide a variety of goods and services vital for human survival such as clean air water food and habitat highlighting the intricate interdependence of all organisms within an ecosystem.",
"Deep learning is a powerful subset of machine learning that uses multi-layered neural networks to mimic the human brain's decision-making process. These deep neural networks are trained on large datasets to identify patterns recognize relationships and make accurate predictions and classifications.The key aspect of deep learning is the use of multiple hidden layers between the input and output layers which allows the network to learn increasingly complex features and representations from the data. As the network processes the data through these layers it can build a hierarchical understanding starting from simple low-level features and progressively learning more abstract high-level concepts",
"Reinforcement learning is a fundamental concept in machine learning that enables artificial intelligence systems to learn through interaction with an environment   aiming to maximize cumulative rewards. Unlike traditional machine learning approaches that rely on fixed datasets   reinforcement learning agents learn from continuous feedback received as they interact with their surroundings. This dynamic process involves key components such as the agent (learner)   environment (external system)   state (current observation)   action (choices available to the agent)   and reward (feedback guiding the learning process). Through trial and error   reinforcement learning agents adapt their strategies based on the consequences of their actions   making it a powerful tool applicable in various fields like robotics   gaming  finance and healthcare. By learning sequences of decisions and balancing exploration with exploitation reinforcement learning agents strive to optimize long-term benefits making them adept at solving complex and dynamic problems autonomously.",
"Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) are both types of artificial neural networks used in deep learning. RNNs are designed to process sequential data by maintaining an internal state that captures information from previous inputs making them effective for tasks like speech recognition and language modeling. However RNNs face challenges with long-term dependencies due to vanishing gradient problems. In contrast LSTMs are an advanced version of RNNs specifically engineered to address these issues. LSTMs incorporate memory cells and gating mechanisms such as forget input and output gates allowing them to retain information over longer sequences and handle long-term dependencies more effectively. The key advantage of LSTMs lies in their ability to remember information for extended periods making them well-suited for tasks like machine translation speech recognition and time series forecasting. The LSTM architecture includes components like a cell input gate output gate and forget gate which collectively enable the network to process and learn from sequential data more efficiently.",
"Large language models (LLMs) are a new class of powerful artificial intelligence systems that have revolutionized natural language processing. These models are trained on vast amounts of text data   allowing them to understand and generate human language with remarkable accuracy and versatility.LLMs are built on advanced deep learning techniques   particularly the transformer architecture   which enables them to capture complex semantic relationships and contextual information. Through self-supervised learning   these models can learn the patterns and structures of language   and then apply that knowledge to a wide range of tasks   such as answering open-ended questions   summarizing text   translating between languages   and even generating original content."
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
    if st.button("Generate Notes"):
        if user_input:
            generated_points = generator.generate_points(user_input)
            for i in generated_points:
                st.markdown(i)
        else:
            st.write("Please enter a sentence.")
