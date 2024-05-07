import nltk

class notes_generator():
    def __init__(self) -> None:
        nltk.download('punkt')

    def extract_important_points(self, summary):
        # Tokenize the summary into sentences
        sentences = nltk.sent_tokenize(summary)
        
        # Rank sentences by length (a simple proxy for importance)
        important_sentences = sorted(sentences, key=len, reverse=True)
        
        # Return the top 5 most important sentences
        return important_sentences[:5]

    def generate_points(self, summary):      
        # Extract important points from the summary
        important_points = self.extract_important_points(summary)
        
        # Print the important points
        return important_points
