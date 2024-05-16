from code_generator_opt import CodeAutocomplete

if __name__ == "__main__":
    prompts = [
        """from torch import nn
        class LSTM(Module):
            def __init__(self, *,
                         n_tokens: int,
                         embedding_size: int,
                         hidden_size: int,
                         n_layers: int):""",
        """import numpy as np
        import torch
        import torch.nn as""",
        "complete this python fucntion def factorial(n):",
    ]
    autocomplete = CodeAutocomplete()
    autocomplete.autocomplete(prompts)
