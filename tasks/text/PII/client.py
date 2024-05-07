from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
from pprint import pprint
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    RecognizerRegistry,
    PatternRecognizer,
    Pattern,
)
from typing import List, Optional, Tuple
from presidio_anonymizer.entities import RecognizerResult, OperatorResult, OperatorConfig
from presidio_anonymizer.operators import Decrypt
from typing import List, Optional, Tuple, Dict
from deanonymizer import InstanceCounterDeanonymizer
from anonymizer import InstanceCounterAnonymizer
from pii import TextAnalyzerService

if __name__ == "__main__":
    
    text = "My name is Don and my phone number is  212-555-5555, gaurav sarkar is working at Intel and lives in Bangalore"

    # Create an instance of TextAnalyzerService with the first model
    
    text_analyzer_service_model1 = TextAnalyzerService(model_choice="obi/deid_roberta_i2b2")
    #text_analyzer_service_model2 = TextAnalyzerService(model_choice="StanfordAIMI/stanford-deidentifier-base")

    # Analyze text with the first model
    entities_model1 = text_analyzer_service_model1.analyze_text(text)
    
    print("Entities found with model   1:", entities_model1)
    
    anonymized_text, req_dict = text_analyzer_service_model1.anonymize_text(text, entities_model1,operator="encrypt")
    
    print("Key Value Mapping",req_dict)
    
    print("Anonymized text:-", anonymized_text.text)

    deanonymized_text = text_analyzer_service_model1.deanonymize_text(anonymized_text)

    print("deanonymized text:-",deanonymized_text.text)


