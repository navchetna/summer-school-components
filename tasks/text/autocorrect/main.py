import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline, AutoTokenizer

class AutoCorrect:
  
    def __init__(self, model_name='deep-learning-analytics/GrammarCorrector', torch_device="cpu"):
        self.model_name = model_name
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir="my_models/")
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir="my_models/").to(self.torch_device)
        grammar_checker_model_id = "textattack/roberta-base-CoLA"
        self.check_model = RobertaForSequenceClassification.from_pretrained(grammar_checker_model_id, cache_dir="my_models/")
        self.grammar_checker_tokenizer = RobertaTokenizer.from_pretrained(grammar_checker_model_id, cache_dir="my_models/")
        
    def check_grammar(self, input_text):
        grammar_checker_pipe = pipeline("text-classification", model=self.check_model, tokenizer=self.grammar_checker_tokenizer)
        result = grammar_checker_pipe(input_text)[0]
        print(f"input text: {input_text}")
        print(f'predicted label: {"contains_errors" if result["label"] == "LABEL_1" else "no errors"}')
        print(f'predicted score: {result["score"] :.2}')
        
        return round(result["score"], 2)
            
    def correct_grammar(self, input_text, num_return_sequences):
        batch = self.tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch, max_length=64, num_beams=num_return_sequences, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text
