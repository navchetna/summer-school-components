import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class CodeAutocomplete:
    def __init__(self, model_name="shibing624/code-autocomplete-gpt2-base"):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir="my_models/")
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="my_models/").to(self.device)

    def autocomplete(self, prompts):
        # Initialize an empty list to store the completions
        completions = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            outputs = self.model.generate(input_ids=input_ids,
                                          max_length=64 + len(prompt),
                                          temperature=1.0,
                                          top_k=50,
                                          top_p=0.95,
                                          repetition_penalty=1.0,
                                          do_sample=True,
                                          num_return_sequences=1,
                                          length_penalty=2.0,
                                          early_stopping=True)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Append the decoded completion to the list
            completions.append(decoded)
            print("=" * 20)  # Keep the separator for clarity
        # Return the list of completions
        return completions
