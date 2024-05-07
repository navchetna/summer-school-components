from transformers import AutoModelWithLMHead, AutoTokenizer

class SQLTranslator:
  
    def __init__(self, model_name="mrm8488/t5-base-finetuned-wikiSQL"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="my_models/")
        self.model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir="my_models/")

    def translate_query(self, query):
        input_text = f"translate English to SQL: {query}"
        features = self.tokenizer([input_text], return_tensors='pt')
        output = self.model.generate(input_ids=features['input_ids'],
                                      attention_mask=features['attention_mask'],
                                      max_new_tokens=120)
        result = self.tokenizer.decode(output[0])
        return result

