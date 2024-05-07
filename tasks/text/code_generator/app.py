import streamlit as st
import pandas as pd
from main import CodeAutocomplete
import time

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
code = CodeAutocomplete()

# Streamlit app
st.title("🤖 Code Generator")

# Description table as a single dictionary
description_table = {
    "Component": "Code Generator",
    "Category":"N/A",
}

description_table2 = {
    "Model": "shibing624/code-autocomplete-gpt2-base",
    "property":"N/A",
}

message = """
component code_generator{
    service autocomplete{
        [in] string text;
        [out] string text;
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
        formatted_prompts = [
    "from torch import nn\n" +
    "class LSTM(Module):\n" +
    "    def __init__(self, *,\n" +
    "                 n_tokens: int,\n" +
    "                 embedding_size: int,\n" +
    "                 hidden_size: int,\n" +
    "                 n_layers: int):\n",
    "import numpy as np\n" +
    "import torch\n" +
    "import torch.nn as\n",
    "def factorial(n):\n",
        ]
        for i in range(warmup_criteria):
            
            code.autocomplete([formatted_prompts])

        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            code.autocomplete([formatted_prompts])                    
            
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

    formatted_prompts = [
        "from torch import nn\n" +
        "class LSTM(Module):\n" +
        "    def __init__(self, *,\n" +
        "                 n_tokens: int,\n" +
        "                 embedding_size: int,\n" +
        "                 hidden_size: int,\n" +
        "                 n_layers: int):\n",
        "import numpy as np\n" +
        "import torch\n" +
        "import torch.nn as\n",
        "def factorial(n):\n",
    ]
    user_input = st.text_area("Enter your code here:", "\n".join(formatted_prompts))
    if st.button("🤖 Complete Code"):
        if user_input:
            # Call the autocomplete function and capture its return value
            answer = code.autocomplete([user_input])

            st.write("Complete Code:")
            st.code(answer[0], language='python')  # Display the output in Streamlit
        else:
            st.write("Please enter a sentence.")
