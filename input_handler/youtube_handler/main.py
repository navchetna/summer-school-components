from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeTranscriptExtractor:
    def __init__(self):
        None
        
    def fetch_transcript(self, video_id):
        self.video_id = video_id
        transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
        return transcript

    def extract_text(self, video_id):
        transcript = self.fetch_transcript(video_id)
        req_str = " ".join([entry["text"] for entry in transcript])
        
        return req_str






    
    
