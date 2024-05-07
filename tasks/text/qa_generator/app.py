import streamlit as st
import pandas as pd
from main import QuestionAnswering
import time
import random

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Set the page to wide layout
st.set_page_config(layout="wide")

# Initialize the profanity masker
solution = QuestionAnswering()

# Streamlit app
st.title("Answer Generator")

# Description table as a single dictionary
description_table = {
    "Component": "Answer Generator",
    "Category":"N/A",
}

description_table2 = {
    "Model": "deepset/roberta-base-squad2",
    "property":"N/A",
    
}

message = """
component QuestionAnswering{
    service answer_question{
        /**
        * Answers a question based on the provided context
        *
        * @param question Question to be answer.
        * @param context The context related to the question.
        */

        [in] string question;
        [in] string context;
        [out] string answer;
        [out] int error_code;
    };
};
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Libraries Used")
st.table(description_table2)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')



performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=100, step=1)
    if st.button("Start Runs", key = "performance_button"):
        # Load the CSV file
        sentences_df = pd.read_csv('sentences.csv') 
        # Extract the required number of sentences for warmup
        warmup_sentences_text = sentences_df['text'].head(warmup_criteria).tolist()
        warmup_sentences_context = sentences_df['context'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for i in range(len(warmup_sentences_text)):
            solution.answer_question(warmup_sentences_text[i], warmup_sentences_context[i])
        
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for _ in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence_text = sentences_df['text'].sample(1).iloc[0]
            sentence_context = sentences_df['context'].sample(1).iloc[0]
            
            # Start the timer for this run
            
            solution.answer_question(sentence_text, sentence_context)
            
            
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

description_table3 = {
    "Why is model conversion important?": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.",
    "What is the importance of data preprocessing in machine learning?": "Data preprocessing is crucial in machine learning as it involves cleaning transforming and organizing data to make it suitable for model training helps improve model accuracy and efficiency by handling missing values scaling features and encoding categorical variables.",
    "How does regularization prevent overfitting in machine learning models?":"Regularization techniques like L1 (Lasso) and L2 (Ridge) help prevent overfitting in machine learning models by adding penalty terms to the loss function this discourages complex model behavior promoting generalization and improving model performance on unseen data.",
    "What is the role of activation functions in neural networks?": "Activation functions introduce non-linearity to neural networks enabling them to learn complex patterns and relationships in data. Functions like ReLU Sigmoid and Tanh help in capturing intricate features controlling the flow of information and improving the network's ability to approximate complex functions.",
    "What is the difference between supervised and unsupervised learning?": "Supervised learning involves training models on labeled data to make predictions whereas unsupervised learning involves discovering patterns and relationships in unlabeled data. Supervised learning is used for tasks like image classification while unsupervised learning is used for tasks like clustering and dimensionality reduction.",
    "How does gradient descent optimize model parameters in machine learning?":"Gradient descent is an optimization algorithm used in machine learning to adjust model parameters based on the gradient of the loss function. It iteratively updates the parameters in the direction of the negative gradient minimizing the loss and improving model performance.",
    "What is the role of hyperparameter tuning in machine learning?": "Hyperparameter tuning involves adjusting model hyperparameters such as learning rate batch size and number of hidden layers to optimize model performance. This process helps in finding the best combination of hyperparameters that result in the best model performance on a given dataset.",
    "How does feature engineering improve machine learning model performance?":"Feature engineering involves creating new features from existing ones to improve model performance. This can include techniques like data transformation feature scaling and dimensionality reduction. Effective feature engineering can lead to better model accuracy robustness and interpretability.",
    "What is the difference between a neural network and a decision tree?": "A neural network is a complex model composed of multiple layers of interconnected nodes (neurons) that process inputs and produce outputs. A decision tree is a simple model that uses a tree-like structure to classify data based on feature values. Neural networks are more powerful but also more complex while decision trees are more interpretable but less accurate."
}
# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    # Add a button to randomly select a sample text
    if st.button("Select Random Sample Text"):
        # Randomly select a sample text from the description_table3
        random_question = random.choice(list(description_table3.keys()))
        random_context = description_table3[random_question]
        
        # Fill the session state variables with the selected sample text
        st.session_state.question_input = random_question
        st.session_state.context_input = random_context
    
    # Create a text input field for the user input and set initial values
    question_input = st.text_input("Enter the question here", key="question", value=st.session_state.get("question_input", ""))
    context_input = st.text_input("Enter the context here", key="context", value=st.session_state.get("context_input", ""))
    
    if st.button("Start Runs"):
        if question_input and context_input:
        # Load the CSV file
        
            answer = solution.answer_question(question_input, context_input)
            st.write("Answer :")
            st.write(answer)
        else:
            st.write("Please enter the question and context")