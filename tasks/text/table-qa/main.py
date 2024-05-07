from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

class TapexQuery:
    
    def __init__(self, model_name="microsoft/tapex-base"):
        self.tokenizer = TapexTokenizer.from_pretrained(model_name, cache_dir="my_models/")
        self.model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir="my_models/")
        
    def query_table(self, table, query):
        encoding = self.tokenizer(table=table, query=query, return_tensors="pt")
        outputs = self.model.generate(**encoding, max_new_tokens=512)
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result