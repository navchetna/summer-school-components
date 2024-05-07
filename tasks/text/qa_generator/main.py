from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch


class QuestionAnswering:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.model_name = model_name
        
        self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2") 
        
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", cache_dir="my_models/")
        
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def answer_question(self, question, context):
        QA_input = {
            'question': question,
            'context': context
        }
        
        # inputs = self.tokenizer(question, context, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = self.model(**inputs)

        # predict_answer_tokens = inputs.input_ids[0]
        # output = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        
        res = self.nlp(QA_input)
        return res['answer']
