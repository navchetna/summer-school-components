import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation
import spacy

nlp = spacy.load("en_core_web_sm")

# data = pd.read_csv("reviews.csv")

text = input("Enter a sentence")


def pre_process(text):
    doc = nlp(text)
    processed_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(processed_tokens)



# data['processed_text'] = data['text'].apply(pre_process)

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(data['processed_text'])
lda = LatentDirichletAllocation(n_components=5)
lda.fit(x)



for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    
    top_words_idx = topic.argsort()[-20:][::-1]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(", ".join(top_words))
    