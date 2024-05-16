import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline, AutoTokenizer
import intel_extension_for_pytorch as ipex
import time

class AutoCorrect:
  
    def __init__(self, model_name='deep-learning-analytics/GrammarCorrector', torch_device="cpu"):
        self.model_name = model_name
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir="my_models/")
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir="my_models/").to(self.torch_device)
        
        self.model = ipex.optimize(self.model, dtype=torch.bfloat16, level='O0')
        sample_text = "Them is enjoying the game"
        batch = self.tokenizer([sample_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(self.torch_device)
        print(batch)
        input_ids = batch['input_ids']
        dummy_attention_mask = torch.ones((1, 1), device=self.torch_device)
        dummy_token_type_ids = torch.zeros((1, 1), device=self.torch_device)
        
        # Trace the model
        self.model.eval()
        self.model = torch.jit.trace(self.model, [input_ids, dummy_attention_mask, dummy_token_type_ids])
        
        
        
        
        
        
        # input_ids = inputs['input_ids']
        # # print(type(input_ids))
        # # print(input_ids.size())
        # attention_masks = inputs['attention_mask']
        # decoder_input_ids = torch.ones_like(input_ids)
        
        # # print(attention_masks)
        # # print(decoder_input_ids)
        
        # # jit_inputs = {
        # #     'input_ids': input_ids,
        # #     'attention_mask': attention_masks,
        # #     'decoder_input_ids': decoder_input_ids
        # # }
        
        # jit_inputs = (input_ids, attention_masks, decoder_input_ids)
       
        
   
        
        # with torch.cpu.amp.autocast(), torch.no_grad():
        #     self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
        #     self.model = torch.jit.freeze(self.model)
        # with torch.no_grad():
        #     # y = self.model(input_ids=input_ids, attention_mask=attention_masks, decoder_input_ids=decoder_input_ids)
        #     # y = self.model(input_ids=input_ids, attention_mask=attention_masks, decoder_input_ids=decoder_input_ids)
        #     # self.model.generate(**inputs, max_length=64, num_beams=3, num_return_sequences=3, temperature=1.5)
        #     y = self.model(jit_inputs)
             
        grammar_checker_model_id = "textattack/roberta-base-CoLA"
        self.check_model = RobertaForSequenceClassification.from_pretrained(grammar_checker_model_id, cache_dir="my_models/")            
        self.grammar_checker_tokenizer = RobertaTokenizer.from_pretrained(grammar_checker_model_id, cache_dir="my_models/")
        
    def check_grammar(self, input_text):
        check_grammar_time = time.time()
        grammar_checker_pipe = pipeline("text-classification", model=self.check_model, tokenizer=self.grammar_checker_tokenizer)
        check_grammar_time_end = time.time() - check_grammar_time
        print("Time taken to check grammar : ", check_grammar_time_end)
        result = grammar_checker_pipe(input_text)[0]
        
        print(f"input text: {input_text}")
        print(f'predicted label: {"contains_errors" if result["label"] == "LABEL_1" else "no errors"}')
        print(f'predicted score: {result["score"] :.2}')
        
        return round(result["score"], 2)
            
    def correct_grammar(self, input_text, num_return_sequences):
        batch = self.tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(self.torch_device)
        correct_grammar_time = time.time()
        translated = self.model.generate(**batch, max_length=64, num_beams=num_return_sequences, num_return_sequences=num_return_sequences, temperature=1.5)
        correct_grammar_time_end = time.time() - correct_grammar_time
        print("Time taken to generate suggestions : ", correct_grammar_time_end)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text
