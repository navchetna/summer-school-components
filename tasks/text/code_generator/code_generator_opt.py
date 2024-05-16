import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from optimum.intel.ipex import IPEXModelForCausalLM
import time
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PlaceholderObserver, QConfig
from intel_extension_for_pytorch.llm import optimize as opt
# import openvino as ov

class CodeAutocomplete:
    def __init__(self, model_name="shibing624/code-autocomplete-gpt2-base"):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir="my_models/")
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="my_models/").to(self.device)

        
        """qconfig = QConfig(activation = PlaceholderObserver.with_args(dtype=torch.bfloat16, is_dynamic=True),
                          weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        
        
        sample_text = """"""import numpy as np 
                        import torch 
                        import torch.nn as nn""""""
                    
        prepared_model = prepare(self.model, qconfig, example_inputs=sample_text, inplace=False)
        self.model.eval()
        self.convert_model = convert(prepared_model)"""
        
        # self.convert_model, self.optimizer = opt(self.model, torch.optim.Adam(self.model.parameters(), lr=0.001),dtype=torch.bfloat16)
        
        # self.convert_model = ipex.optimize(self.model, dtype=torch.bfloat16) #This is not working , data type issues
            
    def autocomplete(self, prompts):
        # Initialize an empty list to store the completions
        total_time = 0
        completions = []
        for prompt in prompts:
            
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            start = time.time()
            outputs = self.convert_model.generate(input_ids=input_ids,
                                          max_length=64 + len(prompt),
                                          temperature=1.0,
                                          top_k=50,
                                          top_p=0.95,
                                          repetition_penalty=1.0,
                                          do_sample=True,
                                          num_return_sequences=1,
                                          length_penalty=2.0,
                                          early_stopping=True)
            time_taken = time.time() - start
            total_time += time_taken
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Append the decoded completion to the list
            print(decoded)
            completions.append(decoded)
            print("=" * 20)  # Keep the separator for clarity
        print("Total Time taken : ", total_time)    
        # Return the list of completions
        return completions
