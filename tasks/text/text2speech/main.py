from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

class Text2Speech():
    def __init__(self):
        self.model = "microsoft/speecht5_tts"
        self.synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[3105]["xvector"]).unsqueeze(0)
    
    def generate_audio(self, prompt = ""):
        speech = self.synthesiser(prompt, forward_params={"speaker_embeddings": self.speaker_embedding})
        
        return speech
