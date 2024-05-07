import streamlit as st
import pandas as pd
from main import TapexQuery
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
query_engine = TapexQuery()

# Streamlit app
st.title("Table Question answer")

# Description table as a single dictionary
description_table = {
    "Component": "Table Question Answering",
    "Category":"N/A",
}

description_table2 = {
    "Library": "microsoft/tapex-base",
    "property":"N/A",
}

message = """
component TapexQuery{
    service query_table{
        /**
        * Querying information from a table
        *
        * @param table The tabular data to be queried.
        * @param analyze_results The question you want to ask about the data in the table.
        * @param result The answer to the query from the table.
        */

        [in] DataFrame table;
        [in] string query;
        [out] string result;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)

# Print the message with the same indentation and format

st.table(description_table2)
st.write("Interface Definition Language (IDL)")
st.code(message, language='plaintext')


data1 = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
data2 = {
    "year": [1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018],
    "city": ["Spain", "Mexico", "Italy", "United States", "France", "South Korea", "Germany", "South Africa", "Brazil", "Russia"]
}
data3 = {
    "year": [1983, 1987, 1992, 1996, 1999, 2003, 2007, 2011, 2015, 2019, 2023],
    "city": ["England", "India", "Australia", "India", "England", "South Africa", "West Indies", "India", "New Zealand", "England", "India"]
}
data4 = {
    "year": [1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022],
    "city": ["Brisbane", "Edinburgh", "Auckland", "Victoria", "Kuala Lumpur", "Manchester", "Melbourne", "Delhi", "Glasgow", "Gold Coast", "Birmingham"]
}



# Performance section
data = [data1, data2, data3, data4]

performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=1, value=5, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=50, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        for i in range(warmup_criteria):
            ind = i % len(data)
            
            table = pd.DataFrame.from_dict(data[ind])

            random_city = random.choice(data[ind]["city"])

            # tapex accepts uncased input since it is pre-trained on the uncased corpus
            query = f"select year where city is equal to {random_city}"
            result = query_engine.query_table(table, query)
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for i in range(runs_criteria):
            ind = i % len(data)
            
            table = pd.DataFrame.from_dict(data[ind])

            random_city = random.choice(data[ind]["city"])

            # tapex accepts uncased input since it is pre-trained on the uncased corpus
            query = f"select year where city is equal to {random_city}"
            result = query_engine.query_table(table, query)
            
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / runs_criteria
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.3f} seconds",
    "Average Time Per Sentence": f"{average_time_per_sentence:.3f} seconds",
}

        st.table(description_table1)



# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    table = 0
    
    if st.button("Select Random Table"):
        # Randomly select a sample text from the description_table3
        random_data = random.choice(data)
        
        table = pd.DataFrame.from_dict(random_data)
        
        random_city = random.choice(random_data["city"])
        
        query = f"select year where city is equal to {random_city}"
        
        # Fill the session state variables with the selected sample text
        st.session_state.question = query
        
    question = st.text_input("Enter your Question", value = st.session_state.get("question", ""))
    table = st.text_input("Enter the table in the form of a Dictionary", value = st.session_state.get("table", ""))
    
    if st.button("Generate answer"):
        if question and table:
            
            result = query_engine.query_table(table, question)
            st.write("Answer from the table")
            st.write(result)
        else:
            st.warning("Please enter a valid Dictionary.")