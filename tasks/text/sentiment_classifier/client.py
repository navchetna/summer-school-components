from main import TextClassifier
import time

if __name__ == "__main__":
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = TextClassifier(checkpoint)
    input_text = """Fraud..!! I paid Rs.500/- for Diesel, however they did not pour even a drop of diesel. Even after refilling the digital cluster was showing same range as before, I thought it will be fine in some time. However it did not get updated. Then I checked what was my run since refuelling - shockingly it was showing my last weeks refuelling details which was 1 week ago. Then I realised that they are frauds.!! """
    start_time = time.perf_counter()
    result = classifier.infer(input_text)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Label: ", result)
    print("Total Time: ", "%.2f" % total_time, " seconds")
