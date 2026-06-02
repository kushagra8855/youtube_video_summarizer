import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline

class YouTubeSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initializes the Hugging Face summarization pipeline.
        We default to a smaller DistilBART model for fast, free local inference.
        """
        print(f"Loading NLP Model: {model_name}...")
        self.summarizer = pipeline("summarization", model=model_name)

    def extract_video_id(self, url: str) -> str:
        """Extracts the unique YouTube video ID from various URL formats."""
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL provided.")

    def fetch_transcript(self, video_id: str) -> str:
        """Fetches and concatenates the video transcript into a single string."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([segment['text'] for segment in transcript_list])
            return full_text
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise RuntimeError(f"Transcript not available for this video. Error: {e}")

    def chunk_text(self, text: str, max_words: int = 400) -> list:
        """
        Splits text into safe chunks to prevent exceeding the model's token limits.
        400 words safely translates to roughly 500-600 tokens.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks

    def summarize_video(self, url: str) -> str:
        """Executes the full data pipeline."""
        print("1. Extracting video ID...")
        video_id = self.extract_video_id(url)
        
        print("2. Fetching transcript data...")
        transcript = self.fetch_transcript(video_id)
        
        print("3. Chunking text for context limits...")
        chunks = self.chunk_text(transcript)
        
        print(f"4. Generating summaries for {len(chunks)} chunk(s)...")
        summaries = []
        for chunk in chunks:
            # Adjust max_length and min_length to control summary density per chunk
            summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            
        final_summary = " ".join(summaries)
        return final_summary

# --- Execution ---
if __name__ == "__main__":
    # Replace with any YouTube URL that has closed captions enabled
    target_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    
    app = YouTubeSummarizer()
    
    try:
        result = app.summarize_video(target_url)
        print("\n--- FINAL SUMMARY ---\n")
        print(result)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
