from langchain.embeddings import HuggingFaceEmbeddings
import json
import sys


class embedder:
    def __init__(self, embedding_model: str = "all-MiniLM-l6-v2", verbose: bool = True):
        self.verboseprint = print if verbose else lambda *a: None
        try:
            self.model_name = embedding_model
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

            self.verboseprint(
                f"EMBEDDER: Embedder initialised successfully with  configuration: embedding_model = {self.model_name}"
            )
        except Exception as e:
            self.verboseprint(
                f"EMBEDDER: Embedder initialization failed. Error:{str(e)}"
            )

    def embed_text(self, text: str) -> str:
        """Generates the embedding for the given text. Text could be any string."""

        try:
            embedding_vector = list(self.embedding_model.embed_query(text))
            return embedding_vector
        except Exception as e:
            self.verboseprint(f"EMBEDDER: embed query  failed. Error:{str(e)}")
