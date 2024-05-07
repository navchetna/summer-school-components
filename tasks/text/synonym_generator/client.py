from main import synonym_generator


if __name__ == "__main__":
    synonyms = synonym_generator()

    # Test the function with a word
    word = "prejudice"
    synonyms = synonyms.find_synonyms(word)
    print("Word : ", word)
    print("Synonyms : ", list(synonyms))
