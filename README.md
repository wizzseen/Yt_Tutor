# Yt_Tutor
This project creates a Retrieval-Augmented Generation (RAG) pipeline that allows users to extract and answer questions from YouTube video transcripts using advanced NLP techniques.

## Features

- **Extract Transcripts from YouTube Videos**  
  Fetch and save transcripts from public YouTube videos effortlessly.

- **Break Down Transcripts into Manageable Chunks**  
  Automatically split lengthy transcripts into smaller, digestible sections for easier processing.

- **Embedding-Based Similarity Search**  
  Leverage embeddings to perform efficient and accurate similarity searches within the transcript.

- **Generate Context-Aware Answers**  
  Use a language model to generate insightful answers based on the provided context.

## Prerequisites
- Python 3.8+
- langchain
- transformers
- youtube_transcript_api
- datasets
- chromadb
- sentence_transformers  



## Installation
1. Clone the repository:
```bash
git clone https://github.com/wizzseen/Yt_Tutor.git
```
2. Create a virtual environment (recommended)
```bash 
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage 

- Provide a YouTube video URL
- Ask a question about the video content
- Receive an AI-generated answer based on the video's transcript 

## Limitations

- Depends on available English transcripts
- Answer quality varies with model and transcript quality

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
