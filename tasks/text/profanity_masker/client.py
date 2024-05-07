from main import profanity_masker

if __name__ == "__main__":
    masker = profanity_masker()
    text = "You're a piece of shit."
    censored_text = masker.mask_words(text)
    print(censored_text)