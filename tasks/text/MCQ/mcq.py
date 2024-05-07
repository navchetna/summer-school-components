from transformers import pipeline


class mcq_generator():
    def __init__(self) -> None:
        # Load the Mistral-7B model and its tokenizer
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.generator = pipeline("text-generation", model=self.model_name)


    def generate_mcqs(self, summary):
        # Prepare the prompt for the model
        prompt = f"<s>[INST]Generate 5 multiple-choice questions with 4 options to each questions with answer and do not mention question number based on the following summary: {summary} and also do not return summary, just return mcqs[/INST]."
        
        # Generate the MCQs
        output = self.generator(prompt, max_length=1000, do_sample=True, temperature=0.8)
        
        # Extract the generated text
        generated_text = output[0]['generated_text']
        
        generated_text = generated_text.replace(prompt, "")
        # Split the generated text into MCQs
        mcqs = generated_text.split("\n")
        
        # Return the MCQs
        return mcqs

        
