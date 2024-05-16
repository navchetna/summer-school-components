import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
import time
from optimum.intel import IPEXModelForMaskedLM


class text_autocomplete():
    def __init__(self) -> None:
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="my_models/")
        self.bert_model = IPEXModelForMaskedLM.from_pretrained("bert-base-uncased", cache_dir="my_models/", export=True).eval()
        
        """
            Tried using ipex.optimize with jit tracing but the issue occurs when using 
            jit tracing since you need sample input for the model optimization.
        """
        

    def decode(self, tokenizer, pred_idx, top_clean):
        ignore_tokens = string.punctuation + "[PAD]"
        tokens = []
        for w in pred_idx:
            token = "".join(tokenizer.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace("##", ""))
        return "\n".join(tokens[:top_clean + 1])

    def encode(self, tokenizer, text_sentence, add_special_tokens=True):
        text_sentence = text_sentence.replace("<mask>", tokenizer.mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if tokenizer.mask_token == text_sentence.split()[-1]:
            text_sentence += " ."

            encoded_dict = tokenizer.encode_plus(
                text_sentence,
                add_special_tokens=add_special_tokens,
                return_tensors='pt',  # Return PyTorch tensors
                return_attention_mask=True,  # Return attention mask
            )

        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        # Find the index of the masked token
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

        return input_ids, attention_mask, mask_idx

    def get_all_predictions(self, text_sentence, top_clean=5):
        # ========================= BERT =================================
        input_ids, attention_masks, mask_idx = self.encode(self.bert_tokenizer, text_sentence)
        token_type_ids = torch.zeros_like(input_ids)
        start = time.time()
        
        with torch.no_grad():
            predict = self.bert_model(input_ids, attention_masks, token_type_ids )[0]
            total_time = time.time() - start
        bert = self.decode(
            self.bert_tokenizer, predict[0, mask_idx, :].topk(top_clean).indices.tolist(), top_clean
        )
        print("Total Time Taken : ", total_time)
        return {"Predictions": bert}


    def get_prediction(self, input_text, top_k):
        try:
            input_text += " <mask>"
            res = self.get_all_predictions(input_text, top_clean=top_k)
            return res
        except Exception as error:
            print("ERROR : ", error)