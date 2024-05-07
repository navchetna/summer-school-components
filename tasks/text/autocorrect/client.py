from main import AutoCorrect

if __name__ == "__main__":
    autocorrect = AutoCorrect()
    text = 'He are moving here.'
    input_text = "They are moved by salar energy"
    
    # Gives a score depending on the grammatical errors in the provided sentence
    print(autocorrect.check_grammar(text))
    
    # Corrects the Grammatical errors found in the sentence
    print(autocorrect.correct_grammar(text, num_return_sequences=2))