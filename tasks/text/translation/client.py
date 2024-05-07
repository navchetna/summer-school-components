from main import TranslationService

def main():
    translator = TranslationService()

    hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
    chinese_text = "生活就像一盒巧克力。"

    # Translate Hindi to French
    translated_hi_text = translator.translate(hi_text, "hi", "fr")
    print(f"Hindi to French: {translated_hi_text}")

    # Translate Chinese to English
    translated_zh_text = translator.translate(chinese_text, "zh", "en")
    print(f"Chinese to English: {translated_zh_text}")

if __name__ == "__main__":
    main()
