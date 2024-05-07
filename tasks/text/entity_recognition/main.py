# from transformers import AutoTokenizer, BertForTokenClassification
# import tensorflow as tf

# class EntityRecognition:
#     def __init__(self) -> None:
#         self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir="my_models/")
#         self.model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir="my_models/")

#     def recognize_entities(self, input_text):
#         inputs = self.tokenizer(input_text, add_special_tokens=False, return_tensors="tf")

#         logits = self.model(**inputs).logits
#         predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

#         # Get the predicted token classes
#         predicted_tokens_classes = [self.model.config.id2label[t] for t in predicted_token_class_ids[0].numpy()]

#         # Get the tokens
#         tokens = self.tokenizer.tokenize(input_text)

#         # Print the predicted token classes and the corresponding tokens
#         for token, predicted_class in zip(tokens, predicted_tokens_classes):
#             print(f"{token}: {predicted_class}")



import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample text for entity recognition
text = "Apple is a company founded by Steve Jobs in California."

# Process the text with spaCy
doc = nlp(text)

# Extract entities from the processed text
entities = [(entity.text, entity.label_, spacy.explain(entity.label_)) for entity in doc.ents]

# Print the entities, their labels, and full forms
for entity, label, explanation in entities:
    print(f"Entity: {entity}, Label: {label} ({explanation})")
