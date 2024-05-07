from transformers import AutoTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name = "facebook/bart-large-cnn") -> None:
        
        self.model_name = model_name
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir="my_models/")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir="my_models/",truncation=True)
        
    def summary_generator(self, input: tuple[str]) -> str :
        inputs = self.tokenizer([input], max_length=1024, return_tensors="pt")
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=512)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return summary
    
