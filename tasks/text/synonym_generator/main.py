from nltk.corpus import wordnet

# Function to find synonyms using NLTK WordNet

class synonym_generator:
    def find_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms
