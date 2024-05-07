from main import YouTubeTranscriptExtractor

if __name__ == "__main__":
    extractor = YouTubeTranscriptExtractor()
    extracted_text = extractor.extract_text("sQuFl0PSoXo")
    print(extracted_text)
