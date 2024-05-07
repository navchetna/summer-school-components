from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class TranslationService:
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir="my_models/")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir="my_models/")

    def translate(self, text, src_lang, target_lang):
        self.tokenizer.src_lang = src_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated_text[0]
