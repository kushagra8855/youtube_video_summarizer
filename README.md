# 📺 YouTube Video Summarizer 

A lightweight, completely free Python pipeline that takes a YouTube video URL and generates a concise text summary. 

This project bypasses the need for paid Google Cloud or OpenAI API keys by extracting auto-generated/manual captions directly from YouTube and running open-source Hugging Face NLP models locally. Originally built as a graduation minor project, this repository represents a streamlined, production-ready rebuild.

## ✨ Features
* **Zero Cost & No API Keys:** Uses `youtube-transcript-api` to scrape captions directly.
* **Local NLP Inference:** Utilizes Hugging Face's `transformers` library (DistilBART) to generate summaries entirely on your machine.
* **Smart Text Chunking:** Automatically breaks long video transcripts into safe token-sized chunks to prevent NLP model memory limits.
* **Object-Oriented Design:** Modular pipeline that makes it easy to swap out scraping methods or upgrade to heavier LLMs later.
